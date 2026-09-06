# Model Math Deep Dive — Current Implementation

Tài liệu này formalize code hiện tại bằng plain text để render ổn định trong GitHub, terminal, VS Code và Colab. Nó mô tả implementation, không phải báo cáo kết quả thực nghiệm.

## 1. Bài toán detection

Một ảnh input sau preprocessing:

```text
I in R^(B x 3 x H x W)
```

FloorPlanCAD có `C=35` class trong benchmark chính. Conditioned detector hỗ trợ hai chế độ.

### Query mode

Mỗi ảnh đi kèm class ID `c_b` và optional text `t_b`:

```text
f(I_b, c_b, t_b) -> (Y_hat_b, S_hat_b, O_hat_b)
```

Shape:

```text
Y_hat in [0,1]^(B x 1 x h x w)
S_hat in R_>0^(B x 2 x h x w)
O_hat in (0,1)^(B x 2 x h x w)
```

### All-class mode

```text
f_all(I) -> (Y_hat_all, S_hat_all, O_hat_all)
```

Shape:

```text
Y_hat_all in [0,1]^(B x C x h x w)
S_hat_all in R_>0^(B x 2C x h x w)
O_hat_all in (0,1)^(B x 2C x h x w)
```

Output stride hiện cố định:

```text
s = 8
h = H / s
w = W / s
```

Do đó `H` và `W` phải chia hết cho 8. Ví dụ `512 x 512 -> 64 x 64`.

## 2. Image encoder

`ConvImageEncoder` là CNN trainable ba lần downsample stride 2:

```text
Z = E_img(I)
Z.shape = [B, C_z, H/8, W/8]
```

Nó không parameterize một probability distribution và không sample latent variable. Tên compatibility `VAEConfig` không thay đổi computation này.

Flatten spatial grid:

```text
N = h * w
Z_tokens = flatten_hw(Z).transpose(channel, token)
Z_tokens.shape = [B, N, C_z]
```

Linear projection tới model dimension `D`:

```text
X_0 = Z_tokens W_img + P_2d
X_0.shape = [B, N, D]
```

`P_2d` là learned positional grid. Khi runtime grid khác preset grid, code reshape `P_2d` về 2D, bilinear-interpolate rồi flatten lại.

## 3. Condition representation

Mọi conditioner trả ba tensor:

```text
T in R^(B x L x D)       # condition tokens
M in {0,1}^(B x L)       # valid-token mask
p in R^(B x D)           # pooled condition
```

Masked mean:

```text
p_b = sum_l M[b,l] * T[b,l] / max(1, sum_l M[b,l])
```

Nếu row không có valid token, numerator bằng 0 và denominator được clamp thành 1, nên pooled output bằng 0 thay vì NaN.

### No condition

```text
T = 0
M = 0
p = 0
```

### Class embedding

Với embedding table `E_cls in R^(C x D)`:

```text
p_b = E_cls[c_b]
T_b = p_b[None, :]
M_b = [1]
```

### Lightweight byte text

Text được encode thành UTF-8 bytes. Byte value `q in {0,...,255}` map tới token ID `q+1`; token ID 0 là padding.

```text
H_l = E_byte[id_l] + E_pos[l]
T_l = LayerNorm(MLP(H_l))
p   = masked_mean(T, M)
```

Embedding/MLP được train từ random initialization. Đây là deterministic tokenization, không phải pretrained language semantics.

### Optional pretrained text

Hugging Face tokenizer/model tạo hidden states `H_hf`; project projection đưa chúng về `D`:

```text
T = LayerNorm(H_hf W_hf)
p = masked_mean(T, attention_mask)
```

Backbone được freeze mặc định. Model/tokenizer chỉ load ở forward đầu tiên và cần optional dependencies cùng weights cache/network access.

## 4. Fusion

Gọi image tokens trước fusion là `X` và pooled condition là `p`.

### None

```text
F = X
```

### Additive

```text
delta = W_add p
F = LayerNorm(X + delta[:,None,:])
```

Với empty condition mask, code nhân `delta` với zero valid-row indicator.

### FiLM

Một MLP sinh `gamma` và `beta`:

```text
[gamma, beta] = MLP_film(p)
F_pre = X * (1 + gamma[:,None,:]) + beta[:,None,:]
F = LayerNorm(F_pre)
```

`gamma=beta=0` tương ứng identity trước output normalization.

### Image-to-condition cross-attention

Current spatial cross-attention dùng image tokens làm query:

```text
Q = LN_img(X) W_Q
K = LN_cond(T) W_K
V = LN_cond(T) W_V

A = softmax(Q K^T / sqrt(d_head) + padding_mask)
C = A V
F = LayerNorm(X + W_o C)
```

Shape theo mỗi head:

```text
Q: [B, heads, N, d_head]
K: [B, heads, L, d_head]
A: [B, heads, N, L]
```

Padding mask loại invalid condition tokens. Empty rows được thay bằng một safe zero token trong attention call, rồi attention delta bị zero sau đó.

### FiLM + cross-attention

```text
X_film = FiLM(X, p)
F = LayerNorm(X_film + CrossAttention(X_film, T, M))
```

### Legacy `current` mode

Compatibility mode đảo query/key direction:

```text
C_text = CrossAttention(query=T, key=X, value=X)
summary = masked_mean(C_text, M)
F = LayerNorm(X + W summary)
```

Mode này tạo global broadcast delta. Nó được giữ để tương thích, không phải default preset contract mới.

## 5. Pathway routing

### Shared pathway

Mọi query dùng cùng parameters:

```text
H_0 = Fusion_shared(X, condition)
H_k = Block_shared_k(H_(k-1))
```

Class differentiation đến từ conditioner/fusion.

### Per-class pathway

Class `c` chọn fusion và block stack riêng:

```text
H_0 = Fusion_c(X, condition)
H_k = Block_(c,k)(H_(k-1))
```

Selected-class batch được partition theo unique class IDs. Với group indices `G_c`:

```text
H_c = Pathway_c(X[G_c])
```

Các output group sau đó được concatenate và permute về batch order ban đầu. Vì routing tự mang class information và thay đổi parameter budget, per-class gains không thể tự động được quy cho text.

## 6. GatedSpatialMixer2D

Input token tensor:

```text
X in R^(B x N x D), N = h*w
```

Projection và split:

```text
[U, G] = X W_in
U, G in R^(B x N x D_inner)
D_inner = expand * D
```

Reshape content về spatial grid:

```text
U_2d = reshape(U) in R^(B x D_inner x h x w)
```

Depthwise convolution:

```text
V_2d[channel] = K_channel (*) U_2d[channel]
```

Mỗi channel có kernel spatial riêng; `groups=D_inner`. Flatten lại và gate:

```text
V = flatten_hw(V_2d)
R = LayerNorm(V) * SiLU(G)
Y = Dropout(R W_out)
```

Block residual stage:

```text
X_1 = X + Mixer(LayerNorm(X))
```

Computation này là gated local 2D mixing. Không có recurrent state, selective scan, state transition matrix hoặc Mamba kernel.

## 7. Self-attention

Từ normalized tokens `X_1`:

```text
[Q, K, V] = X_1 W_qkv
```

Reshape chuẩn:

```text
Q, K, V in R^(B x heads x N x d_head)
d_head = D / heads
```

Attention:

```text
A = softmax(Q K^T / sqrt(d_head), dim=key)
Y = concat_heads(A V) W_o
X_2 = X_1_residual_base + Y
```

Trong block code:

```text
X_2 = X_1 + SelfAttention(LayerNorm(X_1))
```

Attention matrix có `N^2` query-key pairs. Với `N=4096`, mỗi head conceptually có `16,777,216` pairs. PyTorch SDPA có thể dùng memory-efficient kernels, và fallback có thể chunk query dimension, nhưng full-key interaction vẫn giữ computational complexity `O(N^2)`.

## 8. Feed-forward network

```text
FFN(X) = Dropout(Linear_2(Dropout(GELU(Linear_1(X)))))
```

Default expansion:

```text
D -> 4D -> D
```

Residual stage:

```text
X_3 = X_2 + FFN(LayerNorm(X_2))
```

Một `ObjectLearningBlock` hoàn chỉnh:

```text
X_1 = X_0 + Mixer(LN_1(X_0))
X_2 = X_1 + Attention(LN_2(X_1))
X_3 = X_2 + FFN(LN_3(X_2))
```

## 9. Query head

Sau pathway:

```text
H in R^(B x N x D)
H_2d = reshape(LayerNorm(H)) in R^(B x D x h x w)
R = ConvHead(H_2d) in R^(B x 5 x h x w)
```

Channel split:

```text
Y_hat = sigmoid(R[:,0:1])
S_hat = softplus(R[:,1:3])
O_hat = sigmoid(R[:,3:5])
```

Ý nghĩa:

```text
Y_hat[0]   = center confidence
S_hat[0]   = width in output-cell units
S_hat[1]   = height in output-cell units
O_hat[0]   = fractional center x offset
O_hat[1]   = fractional center y offset
```

`softplus` giữ predicted size dương mà không tạo zero-gradient half-space như hard ReLU.

## 10. All-class layout

Với class `c`, regression channels được đặt tại:

```text
size x/y   -> channels 2c, 2c+1
offset x/y -> channels 2c, 2c+1
```

Conditioned shared pathway xử lý class chunks. Với chunk `[a,b)` có `K=b-a` class:

```text
X_expand.shape = [B*K, N, D]
class_ids.shape = [B*K]
```

Sau query head, output reshape về `[B,K,...]` rồi concatenate theo class dimension. Per-class pathway chạy từng class specialist trên cùng encoded image tokens. Baseline trực tiếp xuất `5C` raw channels rồi reshape `[B,C,5,h,w]`.

## 11. Target coordinates

Ground-truth bbox ở input pixels dùng half-open xyxy:

```text
b = (x0, y0, x1, y1)
0 <= x0 < x1 <= W
0 <= y0 < y1 <= H
```

Scale sang output grid:

```text
scale_x = w / W
scale_y = h / H

x0' = x0 * scale_x
y0' = y0 * scale_y
x1' = x1 * scale_x
y1' = y1 * scale_y
```

Size:

```text
bw = x1' - x0'
bh = y1' - y0'
```

Continuous center:

```text
cx_f = (x0' + x1') / 2
cy_f = (y0' + y1') / 2
```

Discrete center cell:

```text
cx = floor(cx_f)
cy = floor(cy_f)
```

Offset target:

```text
dx = cx_f - cx
dy = cy_f - cy
0 <= dx,dy < 1
```

Regression targets ở center cell:

```text
S[0,cy,cx] = bw
S[1,cy,cx] = bh
O[0,cy,cx] = dx
O[1,cy,cx] = dy
M[0,cy,cx] = 1
```

## 12. Center heatmap

Code dùng CornerNet/CenterNet IoU-based radius với `min_overlap=0.7`. Gọi rounded box dimensions là `height=ceil(bh)`, `width=ceil(bw)`; ba candidate radii được tính từ quadratic constraints và lấy minimum, sau đó integer-floor/clamp về radius không âm.

Gaussian kernel diameter:

```text
diameter = 2 * radius + 1
sigma = diameter / 6
```

Kernel:

```text
G(x,y) = exp(-(x^2 + y^2) / (2 sigma^2))
```

Heatmap combine nhiều object bằng elementwise maximum:

```text
Y = max(Y, shifted_G)
```

Sau khi draw, code gán center cell chính xác bằng `1.0`. Điều này quan trọng vì focal loss dùng:

```text
positive = target == 1
```

Target được tạo trực tiếp ở `(h,w)`, không downsample Gaussian từ input resolution.

## 13. Center-cell collisions

Nếu hai bbox cùng query class có center floor vào cùng `(cx,cy)`, một cell không thể lưu hai cặp size/offset khác nhau.

Default policy `largest`:

```text
if cell empty:
    write regression target
elif new_area > stored_area:
    replace stored size/offset
else:
    keep stored size/offset
```

Heatmap vẫn được draw cho từng bbox trước collision resolution. `TargetStats` ghi:

```text
valid_boxes
encoded_boxes
collisions
replacements
ignored_collisions
collision_rate = collisions / valid_boxes
```

Số collision là limitation của representation ở stride 8 và phải được report, không được bỏ qua khi phân tích recall.

## 14. CenterNet focal loss

Cho prediction probability `p` và target heatmap value `y`.

Positive indicator:

```text
I_pos = 1[y == 1]
```

Negative indicator và weight:

```text
I_neg = 1[y < 1]
w_neg = (1 - y)^beta
```

Với defaults `alpha=2`, `beta=4`:

```text
L_pos = log(p) * (1-p)^alpha * I_pos
L_neg = log(1-p) * p^alpha * (1-y)^beta * I_neg
```

Nếu có positives:

```text
L_center = -(sum L_pos + sum L_neg) / N_pos
```

Nếu không có positive, implementation trả negative-loss mean thay vì sum. Normal query dataset được xây từ class thực sự có trong ảnh, nên target contract kỳ vọng có ít nhất một valid center trừ trường hợp annotation/transform bị loại hoàn toàn.

## 15. Masked Smooth L1 losses

Với error `d = prediction - target`, PyTorch Smooth L1 mặc định dùng transition beta 1:

```text
smooth_l1(d) = 0.5 * d^2        if abs(d) < 1
             = abs(d) - 0.5     otherwise
```

Mask một channel được expand sang hai regression channels:

```text
M_2 = expand(M, channels=2)
```

Size loss:

```text
L_size = sum(M_2 * smooth_l1(S_hat - S)) / max(1, sum(M_2))
```

Offset loss:

```text
L_offset = sum(M_2 * smooth_l1(O_hat - O)) / max(1, sum(M_2))
```

Total loss defaults:

```text
L_total = 10 * L_center + 1 * L_size + 1 * L_offset
```

Các weight này là CLI defaults, không phải bằng chứng rằng chúng tối ưu. Chúng phải được ghi trong run config/report.

## 16. Decoder

Sau local max suppression, với peak ở integer output cell `(x,y)`:

```text
center_x_px = (x + O_hat_x[y,x]) * stride_x
center_y_px = (y + O_hat_y[y,x]) * stride_y
width_px    = S_hat_w[y,x] * stride_x
height_px   = S_hat_h[y,x] * stride_y
```

Decoded box:

```text
x0 = center_x_px - width_px / 2
y0 = center_y_px - height_px / 2
x1 = center_x_px + width_px / 2
y1 = center_y_px + height_px / 2
```

Boxes được clip vào input image bounds và loại nếu non-finite, non-positive size hoặc trở thành degenerate sau clip.

`topk` được áp dụng per class sau local-peak selection. Threshold là score floor; AP report phải ghi threshold/top-k vì chúng có thể ảnh hưởng prediction set.

## 17. Detection metrics

Matching diễn ra trong cùng `(image_id, class_id)`:

1. sort predictions theo score giảm dần;
2. với mỗi prediction, chọn unmatched GT có IoU cao nhất;
3. prediction là TP nếu IoU đạt threshold, nếu không là FP;
4. mỗi GT chỉ match tối đa một prediction.

IoU continuous xyxy:

```text
IoU(A,B) = area(intersection(A,B)) / area(union(A,B))
```

Không dùng inclusive-pixel `+1`.

AP dùng 101 recall points:

```text
R = {0.00, 0.01, ..., 1.00}
AP_t = mean_r max(precision where recall >= r)
```

Report chính:

```text
AP50     = class-macro AP at IoU 0.50
AP50:95  = class-macro mean over IoU 0.50,0.55,...,0.95
```

Macro average chỉ gồm class có ít nhất một ground-truth instance. Evaluator hiện không implement COCO area ranges, crowd handling hoặc max-detections variants, nên claim phải gọi đúng metric implementation này thay vì nói chung là full COCO evaluation.

## 18. Validation và held-out test

Training loop dùng query-level validation loss trên `val`, không dùng `test_set`:

```text
train source images -> deterministic image-level train/val split
original test_set   -> untouched test split
```

Checkpoint selection dựa trên `val_loss` hiện tại. Detection AP có thể chạy trên `val` để chọn/tune decoder. Test AP chỉ được chạy sau khi preset, checkpoint, threshold, top-k và hyperparameters đã chốt.

## 19. Checkpoint state

Checkpoint schema v2 chứa:

```text
model_state
model_config + fingerprint
preset
optimizer_state
scheduler_state
epoch + global_step
best_metric + metrics
class mapping + fingerprint
output_stride
split manifest fingerprint
metadata fingerprint
Python/NumPy/PyTorch/DataLoader RNG state
```

Exact resume yêu cầu runtime config và data fingerprints khớp. `--weights-only` chỉ nạp model state rồi restart optimizer/scheduler/RNG.

## 20. Parameter scaling và đo lường

Một số thành phần tuyến tính/attention/FFN scale gần `O(D^2)`, convolutional activation memory scale theo spatial grid, và per-class pathways nhân specialist stacks theo số class. Tuy nhiên tổng parameter count còn phụ thuộc:

- architecture (`floorplan_detector` hoặc baseline);
- shared/per-class routing;
- depth;
- conditioner;
- head/image channels;
- optional pretrained backbone load state.

Vì vậy không dùng bảng estimate cố định. Đo từ model thực tế:

```python
from src.models import build_model

model = build_model("floorplan_base")
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total, trainable)
```

Parameter count không đo FLOPs, latency hoặc memory. Các đại lượng đó phải benchmark riêng trên hardware/input/batch được công bố.

## 21. Falsifiable research questions

Kiến trúc cho phép kiểm tra, nhưng chưa tự trả lời, các câu hỏi:

1. Shared class embedding + FiLM có tốt hơn shared no-conditioning baseline không?
2. Per-class no-text có tốt hơn shared baseline sau khi kiểm soát parameter budget không?
3. Fixed byte text có thêm giá trị ngoài class routing không?
4. Pretrained text có tốt hơn lightweight text trên cùng pathway/fusion không?
5. Cross-attention có tốt hơn FiLM khi giữ các yếu tố khác cố định không?
6. Collision rate ở stride 8 giới hạn recall bao nhiêu trên từng class?

Mỗi kết luận cần cùng metadata v2, split manifest, decoder, metric implementation và measured parameter/latency context. Không được invent AP values hoặc dùng test set để tune rồi gọi đó là held-out result.
