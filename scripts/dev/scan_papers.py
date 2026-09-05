import os

papers = [
    'Detect Anything via Next Point Prediction.pdf',
    'Looking Locally - Object-Centric Vision Transformers as Foundation Models for Efficient Segmentation.pdf',
    'PerceptionDLM - Parallel Region Perception with Multimodal Diffusion Language Models.pdf',
    'VGent -Visual Grounding via Modular Design for Disentangling Reasoning and Prediction.pdf'
]

print("\n" + "="*70)
print("NEW PAPERS IN RESOURCE/")
print("="*70)

for i, paper in enumerate(papers, 1):
    title = paper.replace('.pdf', '')
    size_mb = os.path.getsize(f'Resource/{paper}') / (1024*1024)

    print(f"\n{i}. {title}")
    print(f"   Size: {size_mb:.1f} MB")

    # Infer topic from title
    if 'Next Point' in title or 'Detect Anything' in title:
        print(f"   Topic: Sequential detection, autoregressive decoding")
    elif 'Object-Centric' in title or 'Looking Locally' in title:
        print(f"   Topic: Object-centric ViT, efficient segmentation")
    elif 'Diffusion' in title or 'PerceptionDLM' in title:
        print(f"   Topic: Diffusion models for region perception")
    elif 'Visual Grounding' in title or 'VGent' in title:
        print(f"   Topic: Visual grounding, modular reasoning")

print("\n" + "="*70)
print("RELEVANCE TO FLOORPLANCAD PROJECT")
print("="*70)
print("""
Current: Training DualStreamCenterNet (dual-stream encoder + CenterNet decoder)
Target: AP50:95 >= 0.28 to validate Option B baseline

Most relevant for next phase:
  1. Detect Anything - autoregressive decoding (alternative to CenterNet)
  2. Looking Locally - object-centric ViT (encoder optimization)
  3. PerceptionDLM - parallel region (similar to failed FloorPlanLLM)
  4. VGent - modular reasoning (leverage teacher data)
""")
