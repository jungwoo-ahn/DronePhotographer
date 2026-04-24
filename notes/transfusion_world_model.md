# Transfusion-Style World Model for DronePhotographer

## 1. Motivation

현재 모델: `(image_i, action) → score_json` (composition scores of image_j)

**문제점: Multi-step MPC가 visually blind**

현재 2-step lookahead (`score_with_lookahead_vllm` in `mpc.py:405-480`):
- Step 1: image_i + action_1 → scores_1 (정상)
- Step 2: **같은 image_i** + composed_action(1→2) → scores_2 (문제!)
  - 중간 시각 상태를 모르고 geometric composition만으로 예측
  - Training distribution (distance < 3m) 밖으로 나가면 prediction drift

**해결**: next-frame image_j도 함께 생성하면 multi-step rollout에서 generated frame을 다음 step input으로 사용 가능.

```
Step 0: image_0 + action_1 → scores_1, generated_frame_1
Step 1: generated_frame_1 + action_2 → scores_2, generated_frame_2
Step 2: generated_frame_2 + action_3 → scores_3
```

---

## 2. Architecture Options 비교

| | True Transfusion | VLM + Conditioned Decoder | Discrete Token (Chameleon) |
|---|---|---|---|
| **개념** | 하나의 transformer에서 LM+diffusion 동시 | VLM hidden states → 별도 DiT가 image 생성 | image_j를 VQ-VAE로 discrete token화 후 AR 생성 |
| **파라미터 공유** | 높음 (같은 backbone) | 낮음 (별도 모델) | 높음 (같은 LM head) |
| **구현 복잡도** | 높음 (custom attention, inputs_embeds hack) | 중간 (모듈러) | 낮음 (기존 파이프라인 활용) |
| **생성 품질** | 높음 (continuous latent) | 높음 | 낮음 (quantization loss) |
| **학습 안정성** | 중간 (loss balancing 필요) | 높음 (독립 학습 가능) | 높음 |
| **inference 속도** | 중간 (50-step denoising) | 느림 (별도 모델 forward) | 빠름 (AR generation) |
| **논문 근거** | Transfusion (Meta), λ=5 | Qwen-Image 류 | Chameleon, Emu3 |

**추천: True Transfusion** — 파라미터 공유로 representation 학습 효율 극대화, 논문적 novelty도 있음.

---

## 3. True Transfusion 상세 설계

### 3.1 VAE (Image Tokenizer)

- **모델**: `stabilityai/sd-vae-ft-mse` (Stable Diffusion 1.5 VAE)
- **Target resolution**: 256x256 (원본 1024x768에서 resize)
- **Latent shape**: (4, 32, 32) — downsample factor 8x
- **Patchification**: 2x2 spatial grouping → 16x16 grid = **256 patches**, each dim = 2*2*4 = **16**
- **용도**: image_j의 continuous representation. 학습 중 frozen (never fine-tuned).
- **Pre-caching**: 학습 전 모든 image_j를 encode해서 `.pt`로 저장 (GPU 메모리 절약)

### 3.2 Sequence Format

```
[image_i vision tokens (~225)]  [text prompt (~100)]  [score JSON (~80)]  <BOI>  [image_j patches (256)]  <EOI>
                                                                                     ↑ continuous vectors
Total: ~663 tokens (max_length 2048 내)
```

- `<BOI>`, `<EOI>`: 새로 추가하는 special tokens (discrete)
- image_j patches: continuous float vectors (dim 16), `inputs_embeds`로 주입

### 3.3 Hybrid Attention Mask

```
         text(0:405)  patches(406:661)  eoi(662)
text      causal(▽)      X               X
patches   attend(✓)   bidirectional(■)    X
eoi       attend(✓)     attend(✓)       self
```

- Text tokens: causal (lower-triangular) — 기존과 동일
- Image patches: 서로간 full attention (spatial relationship 학습)
- Image patches → text: attend 가능 (conditioning on text)
- Text → image patches: attend 불가 (미래 정보 유출 방지)

### 3.4 Loss Function

```
L_total = L_LM + λ_diff * L_DDPM + λ_reg * L_regression

L_LM:         Cross-entropy on score JSON tokens (기존)
L_DDPM:       MSE(predicted_noise, actual_noise) at image_j patch positions
L_regression: L1(score_head(hidden), gt_scores) (기존)

λ_diff = 5.0  (Transfusion paper)
λ_reg  = 0.5  (기존 설정)
```

### 3.5 Diffusion Details

- **Noise schedule**: Cosine schedule, T=1000 timesteps
- **Prediction target**: ε-prediction (noise)
- **Timestep embedding**: `MLP(t/T) → hidden_size`, added to each patch embedding
- **Inference**: DDIM sampler, 50 steps (약 1초/image on H200)

### 3.6 Model Modifications

기존 Qwen3.5-2B (hidden_size=2048)에 추가:

```python
# New modules attached to model
model.latent_to_transformer_proj = nn.Linear(16, 2048)    # patch → hidden
model.transformer_to_latent_proj = nn.Linear(2048, 16)    # hidden → patch prediction
model.timestep_mlp = nn.Sequential(
    nn.Linear(1, 2048), nn.SiLU(), nn.Linear(2048, 2048)
)
# Existing
model.score_head = nn.Linear(2048, 13)  # 기존 regression head
```

### 3.7 Training Forward Pass (Pseudo-code)

```python
def compute_loss(model, inputs):
    # 1. Text portion: standard LM
    text_input_ids = inputs["input_ids"]           # [B, text_len]
    text_embeds = model.get_input_embeddings()(text_input_ids)  # [B, text_len, 2048]

    # 2. Image patches: project noisy latents
    noisy_patches = inputs["image_j_noisy_patches"]  # [B, 256, 16]
    t = inputs["diffusion_timesteps"]                 # [B]
    t_embed = model.timestep_mlp(t.unsqueeze(-1) / 1000)  # [B, 2048]
    patch_embeds = model.latent_to_transformer_proj(noisy_patches)  # [B, 256, 2048]
    patch_embeds = patch_embeds + t_embed.unsqueeze(1)  # broadcast timestep

    # 3. Concat: [text_embeds | boi_embed | patch_embeds | eoi_embed]
    inputs_embeds = torch.cat([text_embeds, boi_embed, patch_embeds, eoi_embed], dim=1)

    # 4. Forward with hybrid attention mask
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=hybrid_mask, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]

    # 5. LM loss (text portion only)
    lm_loss = cross_entropy(hidden[:, :text_len], labels[:, :text_len])

    # 6. Diffusion loss (patch portion only)
    patch_hidden = hidden[:, text_len+1 : text_len+1+256]  # skip BOI
    pred_noise = model.transformer_to_latent_proj(patch_hidden)  # [B, 256, 16]
    diffusion_loss = F.mse_loss(pred_noise, noise_target)

    # 7. Regression loss (기존)
    reg_loss = score_regression(hidden, gt_scores)

    return lm_loss + 5.0 * diffusion_loss + 0.5 * reg_loss
```

---

## 4. Implementation Roadmap

나중에 구현할 때 이 순서로:

### Phase 1: VAE Pre-caching (독립, 먼저 가능)
- `src/vlm_qwen25/vae_utils.py` — VAE 로드/encode/decode/patchify 유틸
- `scripts/precache_vae_latents.py` — 전체 이미지 VAE encode → `.pt` 저장
- 검증: decode해서 원본과 비교 (PSNR/SSIM)

### Phase 2: Diffusion Head Module
- `src/vlm_qwen25/diffusion_head.py`
  - `TransfusionDiffusionHead`: projection layers, timestep MLP
  - `build_hybrid_attention_mask()`: 2D attention mask 생성
  - `ddim_sample()`: inference 시 iterative denoising
- 검증: random input으로 forward/backward 확인

### Phase 3: Data Pipeline
- `src/vlm_qwen25/transfusion_dataset.py` — image_j latent 로딩 추가
- `src/vlm_qwen25/transfusion_collator.py` — noisy patch 생성 + hybrid mask 구성
- 검증: collator output shape, attention mask pattern visualization

### Phase 4: Training
- `scripts/train_transfusion.py` — TransfusionTrainer 구현
- `configs/qwen35_vl_2b_transfusion.yaml`
- Training strategy:
  - Warmup (500 steps): projection layers만 학습, backbone frozen
  - Main: 전체 joint training
- 검증: loss 감소, generated image 품질 (FID 또는 visual inspection)

### Phase 5: Inference & MPC Integration
- `mpc.py`에 `generate_next_frame()` 추가
- `mpc_rollout_with_imagined_frames()` — generated frame으로 multi-step rollout
- 검증: 실제 Blender-rendered ground truth와 비교

---

## 5. Key References

| Paper/Repo | 핵심 내용 | 관련 |
|---|---|---|
| [Transfusion](https://arxiv.org/abs/2408.11039) (Meta, 2024.08) | LM + diffusion in one transformer, λ=5, continuous patches | 메인 아키텍처 |
| [Show-o](https://arxiv.org/abs/2408.12528) (NeurIPS 2025) | Discrete diffusion (mask prediction) for images | Alternative |
| [DiffusionVL](https://arxiv.org/abs/2512.15713) | AR VLM을 diffusion으로 fine-tune | Simpler variant |
| [HybridVLA](https://arxiv.org/abs/2503.10631) | Robot VLA with diffusion + AR in one model | Action prediction analogy |
| [lucidrains/transfusion-pytorch](https://github.com/lucidrains/transfusion-pytorch) | PyTorch reference impl (1.3k stars) | 코드 참고 |
| `stabilityai/sd-vae-ft-mse` | SD 1.5 VAE for image tokenization | VAE 사용 |

---

## 6. MPC Integration 구상

### 현재 (score-only, visually blind):
```
image_0 → score(action_1) → score(action_2|same_image_0) → select best 2-step
```

### 목표 (world model, visual rollout):
```
image_0 → [score_1, gen_image_1] → [score_2, gen_image_2] → select best trajectory
```

### Hybrid 전략 (안전):
- **Planning** (lookahead): generated frame 사용 — 빠르고 visual grounding 제공
- **Execution** (actual step): Blender render 사용 — ground truth 보장
- Generated frame 품질이 안 좋으면 score prediction도 부정확해지므로, FID/LPIPS 모니터링 필수

### 예상 속도:
- 1 frame generation: ~1s (50 DDIM steps on H200)
- 3-step lookahead with 30 top-K candidates: ~30s (각 candidate에 대해 frame generation)
- 최적화: top-5만 frame generation, 나머지는 score-only → ~5s

---

## 7. Risks & Open Questions

1. **Joint training이 score prediction quality를 해칠 수 있음**
   - Mitigation: TensorBoard에서 LM loss/regression MAE 개별 모니터링, λ_diff 조정
   
2. **256x256 generated image로 re-scoring이 정확한가?**
   - bbox detection은 resolution-insensitive할 수 있으나 검증 필요
   - Fallback: 384x384로 올리면 patches 576개 (max_length 내)

3. **Qwen의 attention_mask가 2D custom mask를 지원하는가?**
   - SDPA backend: 지원함 (`attn_implementation=None`)
   - Flash Attention 2: arbitrary mask 불가 → flash attention 비활성화 필요
   - 확인 필요: DeepSpeed ZeRO-3와 custom mask 호환성

4. **Compounding error in generated frames**
   - 3-step 이상에서 generated image quality 저하 → score prediction도 drift
   - Mitigation: 2-step까지만 imagination 사용, 그 이상은 score-only

5. **Alternative**: image generation 대신 **latent space에서 직접 scoring**
   - generated image를 decode하지 않고, latent representation에서 바로 score 예측
   - 더 효율적일 수 있으나 설계가 다름 (future work)
