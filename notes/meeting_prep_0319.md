# DronePhotographer 연구미팅 자료 (2026-03-19)

## 1. 프로젝트 개요

DronePhotographer는 드론 카메라의 최적 구도를 자동으로 잡아주는 end-to-end 시스템.

**파이프라인**: Blender 렌더링 → GroundingDINO 검출 → VLM 학습 → MPC 기반 카메라 플래닝

- **Input**: 현재 드론 카메라 이미지 + 카메라 액션 (이동/회전)
- **Output**: 액션 적용 후 예측되는 7개 composition score (JSON)
- **Planning**: MPC로 candidate action 생성 → VLM이 각 action의 score 예측 → 목표에 가장 가까운 action 선택 → 반복

---

## 2. 학습 (Training)

### 2.1 모델 & 방식

| 항목 | 내용 |
|------|------|
| 모델 | Qwen3.5-VL-2B, Qwen3.5-VL-9B, Qwen2.5-VL-7B |
| 학습 방식 | Full parameter fine-tuning (LoRA 아님) |
| Task | 이미지 + 카메라 액션 텍스트 → 7개 score JSON 예측 |
| Loss | Cross-entropy (JSON 토큰만, prompt 토큰은 mask) |
| Precision | bfloat16 mixed precision |

### 2.2 Hyperparameters

| 파라미터 | 값 |
|---------|-----|
| Learning rate | 2.0e-5 |
| Weight decay | 0.01 |
| Warmup | 3% of total steps |
| Epochs | 1 |
| Max sequence length | 2048 tokens |
| Gradient clipping | 1.0 |
| Effective batch size | 16 (모든 설정에서 동일) |

### 2.3 Score Keys (7개)

| Key | 설명 | 범위 |
|-----|------|------|
| `bbox_occupancy_ratio` | 프레임 내 피사체 크기 비율 | [0, 1] |
| `bbox_margin_top` | 상단 여백 | [0, 1] |
| `bbox_margin_bottom` | 하단 여백 | [0, 1] |
| `bbox_margin_left` | 좌측 여백 | [0, 1] |
| `bbox_margin_right` | 우측 여백 | [0, 1] |
| `bbox_aspect_ratio` | bbox 가로/세로 비율 | [0, inf) |
| `bbox_centroid_offset` | 프레임 중심으로부터의 거리 | [0, 1] |

### 2.4 Score Weights (loss 가중치)

```
bbox_occupancy_ratio: 2.0   (2배 중요)
bbox_centroid_offset: 2.0   (2배 중요)
나머지 margin/aspect: 1.0
```

---

## 3. 학습 데이터

### 3.1 렌더링 데이터

| 항목 | 내용 |
|------|------|
| Scene | DogWalk (하얀 눈사람) — **1종류만** |
| Base views | 10,000장 (1024×768 PNG) |
| 카메라 거리 | 2.0 ~ 8.0m |
| 카메라 설정 | DJI Mini 5 Pro 스펙 (focal 24mm, sensor 12.8×9.6mm) |
| 렌더링 시간 | 8000장 기준 ~3시간 50분 (3×A100) |
| Detection 커버리지 | 99.6% (9,959/10,000) |
| 데이터 경로 | `outputs/DogWalk_v2_10k_260309_101152/` |

### 3.2 Training Pair 구성

| 항목 | 값 |
|------|-----|
| Total pairs | **354,728** |
| Train / Eval | 347,634 / 7,094 (98% / 2%) |
| 거리 threshold | 1.5m 이내 nearby view끼리 페어링 |
| Image당 최대 neighbor | 32 |
| Zero-action pairs | 35,473 (10%) — 같은 뷰끼리 페어 |
| Action frame | camera_local (right, up, forward) |
| Rotation 표현 | orientation_6d (forward + up vector) |

### 3.3 Action Text 예시

```
move_camera_local_m(right=0.20, up=-0.10, forward=0.00);
orient_camera_local_6d(fx=0.00, fy=0.10, fz=0.99, ux=0.00, uy=0.99, uz=-0.10)
```

---

## 4. GPU 사용량 & 학습 시간

### 4.1 학습

| 설정 | GPU | Batch (per_device × grad_accum × world_size) | 소요 시간 |
|------|-----|----------------------------------------------|-----------|
| Qwen3.5-2B, 1×H200 | H200 141GB | 8 × 2 × 1 = 16 | **~24시간** |
| Qwen3.5-9B, 2×H200 | H200 141GB ×2 | 1 × 8 × 2 = 16 | 측정 중 |
| Qwen3.5-2B, 4×A100-40G | A100 40GB ×4 | 1 × 4 × 4 = 16 | 측정 중 |

- DeepSpeed Zero-3 사용 (9B, 7B 모델)
- Gradient checkpointing: 9B, 7B만 활성화

### 4.2 렌더링

| 항목 | 값 |
|------|-----|
| GPU | 3× A100-PCIE-40GB (OPTIX) |
| 속도 | ~34 images/min |
| 8000장 | ~3시간 50분 |
| 병목 | 단일 Blender 프로세스가 3 GPU 타일 분할 사용 |
| 개선안 | GPU당 1 Blender 프로세스 → 이론상 ~3배 speedup |

### 4.3 Inference (MPC)

| 항목 | 값 |
|------|-----|
| MPC steps | 16 |
| Candidate actions/step | 최대 720 |
| Candidate batch size | 64~192 |
| Max generation tokens | 128 |

---

## 5. Inference 결과

### 5.1 잘 되는 것

- **Centered framing**: 피사체를 프레임 중앙으로 성공적으로 이동 (margin ~0.5)
- **Distance/size control**: occupancy ratio 조절 가능 (피사체 크기 제어)
- **Rule-of-thirds**: 삼분법 배치 성공
- **기본 composition**: centered + medium distance 조합 안정적

### 5.2 안 되는 것

| 실패 케이스 | 원인 |
|------------|------|
| **Upper-body close-up** | 학습 데이터 2-8m만 존재, 0.5-2m 없음 |
| **Aspect ratio 제어** | 1:1 비율 타겟 설정해도 효과 미미 |
| **3D 추론 부재** | 2D 패턴 암기만 함 (오른쪽 이동 → bbox 왼쪽 이동) |
| **Elevation 제어** | "카메라 낮추고 + 가까이" 같은 복합 3D 동작 불가 |

### 5.3 핵심 분석: 2D 패턴 암기 vs 3D 추론

모델이 학습한 것:
- "카메라 오른쪽 이동 → bbox가 왼쪽으로 이동" (2D 패턴)
- "forward 이동 → occupancy 증가" (단순 관계)

모델이 못하는 것:
- "상반신 클로즈업 = elevation 낮추기 + 접근 + pitch 조절" (3D 기하 추론)
- 카메라-피사체 간 3D spatial relationship 이해

### 5.4 Inference 결과 경로

- `runs/infer_mpc_blender/` — **28개 MPC rollout**
- 각 rollout: `rollout.gif` + `trajectory.json` + `frames/` + `inference_run_info.json`
- 11개 composition preset 테스트 완료

### 5.5 테스트한 Composition Presets

| Preset | 설명 |
|--------|------|
| `centered_50` | 50% 크기, 중앙 |
| `centered_square_medium` | 30% 크기, 1:1 비율, 중앙 |
| `centered_square_close` | 55% 크기, 1:1 비율, 중앙 |
| `centered_wide_18` | 1.8:1 시네마틱 와이드 |
| `cinematic_center_wide` | 35% 크기, 1.8:1, 중앙 |
| `centered_portrait_medium` | 30% 크기, 0.67:1 세로 |
| `centered_portrait_close` | 40% 크기, 0.67:1 세로 |
| `top_right_thirds_medium` | 25% 크기, 1:1, 삼분법 |
| `safe_closeup_portrait` | 클로즈업 시도 (실패) |

---

## 6. 평가 메트릭

### 6.1 Offline Evaluation

- **스크립트**: `scripts/eval_qwen25_vl.py`
- **메트릭**: MAE, RMSE (score key별), Parse Failure Rate
- **샘플 수**: 300 (configurable)
- **Loss 추이** (2B 모델):
  - Step 20: loss = 1.026
  - Step 100: loss = 0.837
  - Step 300: loss = 0.786
  - Step 500: loss = 0.780 → 이후 plateau
- Eval speed: ~4.64 samples/sec

### 6.2 TensorBoard 로그

- `runs/*/checkpoints/runs/Mar*/events.out.tfevents.*`
- Loss, eval_loss, learning_rate, grad_norm 추적

---

## 7. 데이터 관련 이슈 & 계획

### 7.1 현재 한계

| 한계 | 영향 |
|------|------|
| Scene 1종류 (DogWalk) | 일반화 불가 |
| 카메라 거리 2-8m만 | close-up 학습 불가 |
| 2D bbox 메트릭만 | 3D 추론 학습 불가 |
| GroundingDINO 의존 | 렌더링 후 추가 검출 필요 |

### 7.2 해야 할 것

1. **Close-range 렌더링** (0.5-2m) — `--camera_radius_range 0.5 2`
2. **Scene 다양화** — 여러 synthetic room, 다양한 피사체
3. **v3 렌더러 활용** — 이미 구현 완료, mask 기반 ground-truth bbox + 3D metrics
4. **Subject text를 VLM input에 포함** (GroundingDINO prompt 재활용)
5. **Image sampling ratio 결정** — zero-action / small action / large action 비율

### 7.3 render_object_v3 (이미 구현됨)

GroundingDINO 없이 Blender에서 직접 계산하는 새 필드들:

| 필드 | 설명 |
|------|------|
| `elevation_deg` | 카메라-피사체 고도각 |
| `azimuth_deg` | 카메라-피사체 방위각 |
| `camera_subject_distance` | 유클리드 거리 (m) |
| `camera_pitch_deg` | 카메라 pitch 각도 |
| `bbox_2d` | mask 기반 tight bbox (morphological opening) |
| `visibility_ratio` | 가시성 비율 |
| `truncation` | 프레임 밖 잘림 여부 |

---

## 8. 해결 방안 & 연구 방향 (우선순위순)

### P0 — 즉시 실행 가능 (High Impact, Low Effort)

**1. 3D-aware score keys 추가**
- `camera_elevation_angle`, `camera_azimuth_angle`, `camera_subject_distance`, `camera_pitch_angle`
- 기존 annotation 데이터에서 계산 가능 (v3에 이미 구현)
- 수정 파일: `bbox_control.py`, `schema.py`, `dataset.py`, `objective.py`, config yaml

**2. Close-range 데이터 수집 (0.5-2m)**
- 렌더링 파라미터만 변경하면 됨
- 상반신 클로즈업 학습 가능해짐

### P1 — 중기 (High Impact, Medium Effort)

**3. Multi-frame history input**
- 최근 3-5 프레임을 VLM에 함께 입력
- Qwen2.5-VL이 multi-image 지원
- 시간적 맥락 + 암묵적 3D 이해 + oscillation 감소

**4. 새 composition presets**
- `aerial_centered`: 높은 고도, 하향 촬영
- `eye_level_portrait_close`: 눈높이, 가까운 거리
- `heroic_low_angle`: 낮은 각도, 위를 올려다봄

### P2 — 연구 (Medium Impact, Low-Medium Effort)

**5. Test-time CoT reasoning**
- Score 출력 전 step-by-step 추론 강제
- Inference cost ↔ accuracy 트레이드오프

**6. FeedbackDescent**
- Scalar score 대신 pairwise comparison + textual critique
- MPC에서 candidate 비교/정제 반복

### P3 — 장기 연구 (High Impact, High Effort)

**7. SpatialVLM-style 3D reasoning 학습**
- 거리, 고도 QA와 composition 예측 joint training
- 3D 이해 강화

**8. Learned dynamics (VLMPC-style)**
- Blender re-rendering 대신 video prediction model
- Rendering 병목 해소

### P4 — 추후

**9. RL fine-tuning with composition reward**
- Supervised + Blender re-render 기반 reward로 RL
- Prediction → action selection gap 해소

---

## 9. 참고 논문

- VLMPC (RSS 2024) — VLM 기반 MPC 카메라 제어
- SpatialVLM (CVPR 2024) — 3D spatial reasoning VLM
- FeedbackDescent (2025) — Pairwise comparison 기반 최적화
- UP-VLA (2025) — Vision-Language-Action 모델

---

## 10. 핵심 파일 경로

### 노트 & 문서
| 파일 | 내용 |
|------|------|
| `notes/chat_summary_0318.md` | 3/14-18 진행 상황, inference 결과, 한계 분석 |
| `notes/photo_profile_expansion_research.md` | 3D score, FeedbackDescent, 전체 연구 방향 |
| `notes/gpu_optim.txt` | GPU 최적화 방안 |
| `notes/render_speed.txt` | 렌더링 속도 벤치마크 |
| `TODO` | 전체 할 일 목록 |

### 학습 코드
| 파일 | 내용 |
|------|------|
| `scripts/train.py` | 학습 메인 (HuggingFace Trainer) |
| `configs/*.yaml` | 학습 설정 7종 |
| `src/vlm_qwen25/dataset.py` | Training pair 구성 |
| `src/vlm_qwen25/collator.py` | Batch collation |
| `src/vlm_qwen25/prompt.py` | VLM 프롬프트 포맷 |
| `src/vlm_qwen25/rotation_utils.py` | 3D rotation 수학 |

### Inference 코드
| 파일 | 내용 |
|------|------|
| `scripts/infer_mpc_blender.py` | Blender-in-the-loop MPC (700줄) |
| `scripts/infer_mpc.py` | Offline MPC (pre-rendered views) |
| `scripts/infer_mpc_*.sh` | 17개 preset launcher |
| `src/vlm_qwen25/mpc.py` | MPC 엔진 (387줄) |
| `src/vlm_qwen25/objective.py` | Composition 목표 & 에러 계산 (265줄) |

### 평가
| 파일 | 내용 |
|------|------|
| `scripts/eval_qwen25_vl.py` | MAE/RMSE 평가 |
| `src/scoring/bbox_control.py` | Rule-based score 계산 |

### 렌더링
| 파일 | 내용 |
|------|------|
| `render_object.py` | Blender 렌더링 v2 |
| `render_object_v3.py` | v3: mask bbox + 3D metrics |
| `render_object.sh` / `render_object_v3.sh` | Multi-GPU launcher |

### 데이터 & 결과
| 경로 | 내용 |
|------|------|
| `outputs/DogWalk_v2_10k_260309_101152/` | 10k 렌더 데이터셋 (11.4GB) |
| `runs/20260312_150649_qwen35_vl_2b_1xh200/` | 2B 학습 run (47 checkpoints) |
| `runs/infer_mpc_blender/` | 28개 MPC rollout 결과 |
| `docs/figures/` | Pipeline 다이어그램 (SVG) |

---

## 11. 미팅에서 보여줄 것

1. **Pipeline 다이어그램** — `docs/figures/dronephotographer_pipeline.svg`
2. **학습 스펙 요약** — 모델, 데이터, GPU, 시간
3. **Inference 데모** — `runs/infer_mpc_blender/*/rollout.gif` (성공/실패 케이스)
4. **Error reduction graph** — trajectory.json에서 step별 error 추이
5. **한계 분석** — 2D 패턴 암기 문제, close-up 실패 원인
6. **Next steps** — P0 (3D scores + close-range data) 즉시 실행 계획
