# Model Math Deep Dive — Render-Safe Version

> Phiên bản này tránh dùng LaTeX `\[...\]` vì một số markdown viewer không render được. Công thức được viết bằng plain text / code block để đọc được ở GitHub, terminal, VS Code, Colab hoặc viewer thường.

---

## 1. Formalize bài toán

Với một ảnh floor plan:

```text
I ∈ R^(3 × H × W)
```

Với 35 class:

```text
c ∈ {0, 1, ..., 34}
```

Mỗi class có một text prompt cố định:

```text
T_c = "Find {class_name} in this floor plan drawing"
```

Model học ánh xạ:

```text
f_θ(I, T_c, c) → (Y_hat_c, S_hat_c, O_hat_c)
```

Trong đó:

```text
Y_hat_c ∈ [0, 1]^(1 × h × w)       # center heatmap cho class c
S_hat_c ∈ R_>=0^(2 × h × w)        # size map: width, height tại tâm object
O_hat_c ∈ [0, 1)^(2 × h × w)       # offset map: fractional dx, dy trong output cell
```

Vì VAE/downsample factor = 8:

```text
h = H / 8
w = W / 8
```

Với ảnh 512×512:

```text
h = w = 64
```

---

## 2. Triết lý kiến trúc

Dự án không phải object detection thông thường. Concept chính là **Conditioned Reflex Learning**:

- **Text prompt** là tác nhân kích thích — stimulus.
- **Mỗi class có pathway riêng** — giống một đường dây thần kinh chuyên biệt.
- Model không tự detect mọi thứ một cách mù; nó phản ứng với stimulus.

Ví dụ:

```text
"Find chair" → kích hoạt pathway chair → output vị trí chair
"Find wall"  → kích hoạt pathway wall  → output vị trí wall
```

Pipeline ý tưởng:

```text
Image
  ↓
VAE Encoder
  ↓
Image tokens
  ↓
For class c:
  Fixed text T_c
      ↓
  Text Encoder
      ↓
  Early Fusion(image tokens, text tokens)
      ↓
  class_blocks[c]
      ↓
  CenterNet Head
      ↓
  center_heatmap + size_map + offset_map
```

---

## 3. Image encoder: ảnh thành token

Input batch:

```text
I.shape = [B, 3, 512, 512]
```

Qua VAE encoder stub:

```text
Z = E_vae(I)
Z.shape = [B, 16, 64, 64]
```

Flatten spatial dimension:

```text
64 × 64 = 4096
Z_flat.shape = [B, 4096, 16]
```

Project latent channel 16 sang model dimension D:

```text
X = Z_flat @ W_img
X.shape = [B, 4096, D]
```

Với `model_dim=256`:

```text
X.shape = [B, 4096, 256]
```

Mỗi token tương ứng với một vùng 8×8 pixel trên ảnh gốc.

---

## 4. Text encoder: stimulus thành embedding

Với class `chair`:

```text
T_c = "Find chair in this floor plan drawing"
```

Tokenized thành:

```text
ids_c.shape = [L]
L = 32 hiện tại
```

Text encoder stub tạo embedding kiểu T5:

```text
U_c.shape = [B, L, 4096]
```

Project xuống model dimension:

```text
Q_c = U_c @ W_text
Q_c.shape = [B, L, D]
```

Ý nghĩa:

- Text không phải phụ kiện.
- Text là **stimulus chính**.
- Mỗi text class định hướng model tìm đúng object tương ứng.

---

## 5. Early Fusion: text hỏi ảnh

EarlyFusion hiện tại dùng cross-attention với:

```text
Query = text tokens
Key   = image tokens
Value = image tokens
```

Attention chuẩn:

```text
Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d)) @ V
```

Ở đây:

```text
Q = Q_c
Q.shape = [B, L, D]

K = V = X
K.shape = V.shape = [B, N, D]

N = 4096
```

Attention matrix:

```text
A_c = softmax((Q_c @ X^T) / sqrt(d))
A_c.shape = [B, L, N]
```

Output:

```text
F_c = A_c @ X
F_c.shape = [B, L, D]
```

Code hiện tại mean-pool text-aware output:

```text
F_bar_c = mean(F_c, dim=text_length)
F_bar_c.shape = [B, 1, D]
```

Broadcast về toàn bộ image tokens:

```text
X'_c = X + Linear(F_bar_c)
X'_c.shape = [B, N, D]
```

### Nhận xét

EarlyFusion hiện tại tạo một **global text-aware bias** cho toàn bộ ảnh. Mọi spatial token nhận cùng một vector điều kiện hóa bởi text.

Ưu điểm:

- Đơn giản
- Rẻ
- Dễ train

Nhược điểm:

- Spatial selectivity còn yếu
- Text chưa trực tiếp modulate từng token ảnh

Một hướng nâng cấp sau này:

```text
X'_c = X + CrossAttention(Q=image_tokens, K=text_tokens, V=text_tokens)
```

Khi đó mỗi image token tự hỏi: “với text này, tôi có liên quan không?”

---

## 6. Per-class Object Learning Blocks

Sau fusion, feature được route vào block riêng của class:

```text
X'_c → B_c
```

Có 35 pathways:

```text
B_0, B_1, ..., B_34
```

Mỗi `B_c` là stack gồm `depth_per_class` ObjectLearningBlocks.

Với depth = 2:

```text
H_c^(0) = X'_c
H_c^(1) = OLB_c^(0)(H_c^(0))
H_c^(2) = OLB_c^(1)(H_c^(1))

H_c = H_c^(2)
```

Ý nghĩa sinh học:

```text
class_blocks[4]  → chair pathway
class_blocks[8]  → door_double pathway
class_blocks[30] → wall pathway
```

Mỗi block tích lũy kinh nghiệm riêng cho class của nó.

---

## 7. ObjectLearningBlock toán học

Một ObjectLearningBlock gồm 3 residual stages:

```text
x = x + Mamba(LayerNorm(x))
x = x + SelfAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Viết đầy đủ:

```text
X1 = X  + Mamba(LN(X))
X2 = X1 + Attention(LN(X1))
X3 = X2 + FFN(LN(X2))

OLB(X) = X3
```

Đây là kiến trúc **Pre-Norm Residual**, giống nhiều Transformer hiện đại.

---

## 8. Mamba-like block hiện tại

MambaBlock hiện tại là approximation, chưa phải selective scan thật.

Input:

```text
X.shape = [B, N, D]
```

Linear projection:

```text
[X_ssm, Z] = X @ W_in
X_ssm.shape = Z.shape = [B, N, d_inner]
```

Depthwise Conv1D theo sequence:

```text
C = DWConv1D(X_ssm)
```

Activation:

```text
C' = SiLU(C)
```

Gating:

```text
G = LayerNorm(C') * SiLU(Z)
```

Project về D:

```text
Y = G @ W_out
Y.shape = [B, N, D]
```

### Nhận xét

Token order hiện tại là flatten 2D sang 1D:

```text
(0,0), (0,1), ..., (0,63),
(1,0), (1,1), ...
```

Do đó Conv1D học tốt quan hệ gần theo raster order. Quan hệ dọc trong ảnh cách nhau 64 token.

Mamba thật với selective scan sẽ phù hợp hơn cho long-range dependencies.

---

## 9. Self-Attention trong OLB

Self-attention chuẩn:

```text
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

A = softmax((Q @ K^T) / sqrt(d))
Y = A @ V
```

Với `N = 4096`:

```text
A.shape = [4096, 4096]
4096^2 = 16,777,216 attention pairs
```

Self-attention giúp model học global relations:

- door nằm trên wall
- chair gần table
- sink/toilet trong bathroom
- wall và dimension line trải dài toàn ảnh

Nhược điểm là chi phí tính toán cao, đặc biệt khi inference 35 classes.

---

## 10. FFN

FFN có dạng:

```text
FFN(X) = W2 * activation(W1 * X)
```

Với expansion 4:

```text
D → 4D → D
```

Với `D=256`:

```text
256 → 1024 → 256
```

FFN chịu trách nhiệm nonlinear feature mixing sau khi sequence/spatial information đã được Mamba và Attention trộn.

---

## 11. CenterNet Head

Sau class block:

```text
H_c.shape = [B, 4096, D]
```

Reshape về spatial map:

```text
H_2d.shape = [B, D, 64, 64]
```

Qua Conv head:

```text
O_c = ConvHead(H_2d)
O_c.shape = [B, 5, 64, 64]
```

Tách channels:

```text
center_heatmap = sigmoid(O_c[:, 0:1])
size_map       = relu(O_c[:, 1:3])
offset_map     = sigmoid(O_c[:, 3:5])
```

Trong đó:

```text
center_heatmap: xác suất tâm object
size_map[0]   : width
size_map[1]   : height
```

---

## 12. CenterNet target

Với bbox:

```text
bbox = (x0, y0, x1, y1)
```

Scale từ ảnh gốc 512 sang output 64:

```text
s = 64 / 512 = 1 / 8

x0' = x0 * s
y0' = y0 * s
x1' = x1 * s
y1' = y1 * s
```

Center:

```text
cx = (x0' + x1') / 2
cy = (y0' + y1') / 2
```

Size:

```text
w = x1' - x0'
h = y1' - y0'
```

Heatmap Gaussian:

```text
Y[x, y] = exp(-((x - cx)^2 + (y - cy)^2) / (2 * sigma^2))
```

Nếu nhiều object overlap:

```text
Y = max(Y, Gaussian_for_current_bbox)
```

Size map chỉ có giá trị tại center:

```text
S[0, cy, cx] = w
S[1, cy, cx] = h
M[cy, cx]    = 1
```

---

## 13. Focal Loss

Prediction:

```text
p = predicted center_heatmap value
```

Target:

```text
y = target center_heatmap value
```

Positive pixel:

```text
y = 1
```

Negative pixel:

```text
y < 1
```

Positive loss:

```text
L_pos = log(p) * (1 - p)^alpha
```

Negative loss:

```text
L_neg = log(1 - p) * p^alpha * (1 - y)^beta
```

Total:

```text
L_focal = -1/N * (sum(L_pos over positives) + sum(L_neg over negatives))
```

Default:

```text
alpha = 2
beta  = 4
```

Ý nghĩa:

- Tâm thật phải có `p → 1`
- Background phải có `p → 0`
- Vùng gần tâm Gaussian không bị phạt quá nặng nhờ `(1-y)^beta`

Ví dụ nếu `y = 0.7`:

```text
(1 - y)^4 = 0.3^4 = 0.0081
```

---

## 14. Masked L1 Size Loss

Prediction:

```text
S_hat.shape = [B, 2, h, w]
```

Target:

```text
S.shape = [B, 2, h, w]
```

Mask:

```text
M.shape = [B, 1, h, w]
M ∈ {0, 1}
```

Loss:

```text
L_size = sum(M * abs(S_hat - S)) / (sum(M) + eps)
```

Chỉ tính tại center object.

---

## 15. Total Loss sau chỉnh sửa

Trước đây:

```text
L = 1.0 * L_focal + 0.1 * L_size
```

Sau khi training log cho thấy focal collapse, chỉnh thành:

```text
L = 10.0 * L_focal + 0.1 * L_size
```

Lý do:

```text
Epoch 2:
focal  ≈ 0.0000
size_l1 ≈ 22
total  ≈ 2.2
```

Vì:

```text
0.1 * 22 = 2.2
```

Focal gần như biến mất, model không còn đủ áp lực học center. Tăng focal weight giúp ép model tiếp tục học vị trí tâm object.

---

## 16. Vì sao focal loss có thể collapse?

Center heatmap rất sparse.

Output 64×64 có:

```text
4096 pixels
```

Một ảnh/class có thể chỉ có 1–5 centers.

Nếu model predict gần 0 mọi nơi:

```text
p ≈ 0
```

Negative loss:

```text
log(1-p) * p^alpha ≈ 0
```

Positive loss đáng lẽ phải lớn, nhưng code hiện tại xác định positive bằng:

```python
pos_inds = target.eq(1).float()
```

Nếu target được tạo ở 512×512 rồi downsample về 64×64 bằng bilinear, peak Gaussian có thể không còn đúng bằng 1 nữa. Khi đó:

```text
num_pos = 0
```

và code rơi vào nhánh chỉ tính negative loss:

```python
if num_pos == 0:
    return -neg_loss.sum()
```

Khi model all-zero, negative loss gần 0, focal collapse.

### Điểm cần kiểm tra kỹ

Nếu log vẫn tiếp tục có:

```text
focal=0.0000
```

thì nhiều khả năng cần sửa target generation:

1. Generate CenterNet target trực tiếp ở output resolution 64×64
2. Hoặc đổi positive condition từ `target.eq(1)` sang threshold, ví dụ `target.ge(0.99)`

Cách đúng hơn là target nên được sinh thẳng ở output stride.

---

## 17. Warmup schedule

Sau khi chỉnh:

```text
lr_max = 1e-5
warmup_steps = 500
warmup_start_factor = 0.1
```

Step đầu:

```text
lr_0 = 1e-6
```

Step 500:

```text
lr_500 = 1e-5
```

Warmup tuyến tính:

```text
lr_t = lr_max * (start_factor + (1 - start_factor) * t / warmup_steps)
```

Sau warmup dùng cosine decay:

```text
lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

Với:

```text
lr_min = 0.01 * lr_max = 1e-7
```

---

## 18. Parameter scaling

Tham số trong OLB chủ yếu scale theo:

```text
O(D^2)
```

Do đó khi tăng `model_dim` từ 256 lên 512:

```text
(512 / 256)^2 = 4
```

Số tham số trong OLB tăng khoảng 4 lần.

Theo model docs:

| Component | model_dim=256 | model_dim=512 |
|---|---:|---:|
| VAE stub | ~0.66M | ~0.66M |
| Text stub | ~8.4M | ~16.8M |
| 35 × OLB × depth 2 | ~119.8M | ~479M |
| Total | ~131M | ~500M+ |

Vì vậy `model_dim=512` rất nặng. `model_dim=256` hợp lý hơn cho baseline.

---

## 19. Training semantics

Dataset đã expand thành các cặp:

```text
(image_1, chair)
(image_1, door_double)
(image_1, table)
(image_2, wall)
...
```

Mỗi sample có:

```text
(I_i, c_i)
```

Model chạy:

```text
X_i = E_vae(I_i)
H_i = B_c_i(Fusion(X_i, T_c_i))
```

Loss:

```text
L_i = Loss(f(I_i, T_c_i, c_i), target_for_class_c_i)
```

Gradient update:

- Shared image encoder
- Shared text encoder
- Early fusion của class đó
- Pathway/block của class đó
- Shared CenterNet head

Vì dataset expanded nên trong một epoch, các class xuất hiện trong toàn bộ train set đều được train.

---

## 20. Inference semantics

Input inference chỉ có một ảnh:

```text
I
```

Model built-in sẵn 35 texts:

```text
T_0, T_1, ..., T_34
```

Inference all-class:

```text
for c in 0..34:
    Y_hat_c, S_hat_c = f(I, T_c, c)
```

Sau đó decode:

1. Threshold heatmap
2. Local max / NMS
3. Lấy center `(x, y)`
4. Lấy size `(w, h)`
5. Convert thành bbox:

```text
x0 = x - w / 2
y0 = y - h / 2
x1 = x + w / 2
y1 = y + h / 2
```

Scale về ảnh gốc:

```text
x_orig = 8 * x
y_orig = 8 * y
w_orig = 8 * w
h_orig = 8 * h
```

---

## 21. Điểm yếu hiện tại

### 21.1 EarlyFusion spatial selectivity yếu

Hiện tại text-aware feature bị mean-pool rồi broadcast toàn ảnh. Điều này tạo global conditioning nhưng chưa tạo spatial attention mạnh.

Có thể nâng cấp bằng FiLM:

```text
gamma_c, beta_c = MLP(mean_text_embedding)
X'_c = gamma_c * X + beta_c
```

Hoặc image-query cross-attention:

```text
X'_c = X + CrossAttention(Q=X, K=T_c, V=T_c)
```

### 21.2 MambaBlock chưa phải Mamba thật

Block hiện tại là approximation bằng Conv1D + gating. Để có long-range modeling tốt hơn, production nên thay bằng:

```python
from mamba_ssm import Mamba
```

### 21.3 Size scale cần kiểm chứng

Cần đảm bảo size target và decode cùng scale.

Khuyến nghị:

- Target heatmap nên ở output resolution 64×64
- Size map nên lưu width/height cũng ở output resolution
- Khi decode về ảnh gốc thì nhân stride 8

---

## 22. Checklist khi train lại

Sau khi chỉnh LR/focal/warmup, log tốt nên có dạng:

```text
Epoch 1:
- focal giảm nhưng không biến mất quá sớm
- size_l1 giảm nhẹ
- val_loss giảm

Epoch 2:
- focal vẫn còn meaningful, không nên toàn 0.0000
- val_loss không tăng mạnh
```

Nếu vẫn thấy:

```text
focal=0.0000
```

thì ưu tiên debug target:

1. Kiểm tra `target.eq(1).sum()` sau downsample
2. Nếu bằng 0 nhiều batch → sinh target trực tiếp ở output resolution
3. Hoặc tạm đổi positive mask sang `target.ge(0.99)`

---

## 23. Công thức tổng kết

Model:

```text
f_θ(I, T_c, c)
  = Head(
      B_c(
        Fusion(
          E_vae(I),
          E_text(T_c)
        )
      )
    )
```

Loss:

```text
L = 10 * L_focal(Y_hat_c, Y_c)
  + 0.1 * L_size(S_hat_c, S_c, M_c)
```

Core idea:

> Text class là stimulus. 35 class pathways là các phản xạ chuyên biệt. CenterNet head chuyển phản xạ đó thành tâm object và kích thước bbox.
