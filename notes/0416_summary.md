# 2026-04-16 작업 요약

## 0. 환경 세팅 + 의존성 설치
- conda env `ahn` (Python 3.11)에 requirements.txt 전체 설치 (PyTorch 2.11, transformers 5.5 등)
- Blender 4.5.0 portable 다운로드 설치 (`blender/blender`, `scripts/install_blender.sh`)
- `outputs/`, `runs/` 데이터 로컬 이동, `run_info.json` 경로 수정 (nas5 → davianlab)
- flash-attn 2.8.3 빌드 설치 → 이후 vLLM과 torch 버전 충돌로 제거

## 1. Generate Target 모듈 테스트 + E2E 파이프라인 구축

### generate_target.py 테스트
- `--backend claude-cli` 로 자연어 → target JSON 변환 확인
- 4개 다양한 prompt로 생성 테스트 (front eye-level, left hero, high angle, right cinematic)
- **발견된 문제**: occupancy=0.6 + aspect_ratio=0.7 같은 프레임 초과 조합 생성 → crash
- **발견된 문제**: c2o orthogonality 제약과 의미적 일관성 충돌 (하이앵글에서 uz<0)
- **발견된 문제**: forward 벡터 norm이 1 미달인 경우 (|f|=0.85)

### E2E 스크립트 생성
- `scripts/run_e2e_mpc.sh`: 자연어 → generate_target (claude-cli) → target JSON 추출 (jq) → infer_mpc_blender
- CUDA_VISIBLE_DEVICES, RUN_DIR, MODEL_PATH 등 환경변수 override 가능
- TARGET_JSON 직접 지정으로 Stage 1 스킵 가능

## 2. VLM Inference 최적화

### vLLM 통합
- `score_action_candidates_vllm()` 함수 추가 (`mpc.py`)
- `--use_vllm`, `--tensor_parallel_size` 플래그 추가 (`infer_mpc_blender.py`, `infer_mpc.py`)
- PagedAttention + continuous batching → 720 candidates ~9초 처리 (H200 1장)
- tokenizer 호환성 수정 (TokenizersBackend → Qwen2TokenizerFast)
- vLLM 설치 시 transformers 5.5→4.57 다운그레이드, torch 2.11→2.10 → flash-attn ABI 충돌 → flash-attn 제거 (vLLM 자체 flashinfer 사용)

### non-vLLM 경로
- `attn_implementation="flash_attention_2"` 추가
- `torch.compile(model, mode="reduce-overhead")` 추가

## 3. MPC Rollout 실험 (v1: 기존 모델, safety guard 없음)

### 실험 설정
- 모델: `qwen35_vl_2b_1xh200_with_c2o_5k` (기존 checkpoint)
- GPU 0~3 병렬, 4개 target × 50 steps, Blender-in-the-loop
- Prompts: front eye-level / left hero / high angle thirds / right cinematic

### 결과
| Prompt | Init→Final Error | 비고 |
|--------|:---:|------|
| front eye-level | 0.18→0.07 | 가장 잘됨, 하지만 중간에 error 0.30 스파이크 반복 |
| left hero | 0.35→0.28 | 느린 수렴 |
| high angle thirds | 0.40→0.39 | 거의 수렴 안됨 |
| right cinematic | 0.38→0.28 | 느린 수렴 |

### 발견된 문제
- **사람이 프레임 밖으로 나가면 모델 예측 붕괴** (c2o 값 fy=-0.431 반복 출력)
- Step 10에서 사람 잘 보이다가 → Step 17에서 완전히 사라짐 → error 0.30으로 스파이크
- 카메라가 너무 가까이 접근 + 아래로 drift

## 4. MPC Safety Guards 적용 (v3)

### 적용한 3가지
1. **c2o spike detection**: 이전 step 대비 c2o 급변 시 최소 이동 선택
2. **margin 하한 필터**: margin_top/bottom < 0.03이면 candidate 제외
3. **adaptive step size**: 거리 비례 스케일링 (가까울수록 작은 step)

### 2-step lookahead MPC 구현
- top-30 1st action에서 각각 follow-up 50개 생성 → 기하학적 합성 → 재랭킹
- `score_with_lookahead_vllm()` 함수, `--lookahead_top_k 30` 플래그

### v3 결과 (safety guards + lookahead)
- error 스파이크 완전 제거 (v1: 0.30 스파이크 → v3: 없음)
- 하지만 수렴 속도/최종 error는 비슷하거나 일부 악화

### 코드 버그 발견 및 수정
- spike detection: `max()` empty sequence crash → `default=0.0`
- lookahead re-ranking: objective 스케일 불일치 → 전체 통일
- margin filter: non-bbox 모델에서 no-op → score_keys 체크
- lookahead: world frame 좌표계 불일치 → 분기 처리

## 5. 총 12개 실험 (Batch 1~3) 완료

Batch 1 (비교용 동일 target) + Batch 2 (새 target 4개) + Batch 3 (새 target 4개)
- >50% 감소: 5/12
- 15~50% 감소: 5/12  
- 악화: 1/12 (left hero below)
- 최고: front close-up 0.164→0.050 (69.5% 감소)

## 6. 근본적 문제 분석

### Inference step size 문제
- 학습 데이터: pair 거리 중간값 1.13m (threshold 1.5m)
- 추론 step: max 0.3m → 학습 분포의 20%만 사용
- 모델은 1m+ 이동도 예측 가능하지만 inference에서 안 쓰고 있었음

### Occupancy 함정
- occupancy = 보이는 bbox 면적 / 이미지 면적 (mask 기반)
- 사람이 잘리면 occupancy 감소 → MPC가 "더 가까이" 선택 → 악순환
- occupancy만으로는 "잘 보이는 상태"와 "잘린 상태" 구분 불가

### 카메라 drift
- MPC greedy 특성: 매 step 단기 이득만 추구
- 50 step 누적 → eye-level target인데 low-angle로 변함

## 7. 학습 방법 개선점 발견

1. **데이터 부족**: 5k images에서 17.6k pairs → 2B 모델치고 적음
2. **1 epoch만 학습**: 부족. 3~5 epoch 권장
3. **Loss 설계**: cross-entropy on JSON tokens → 수치 회귀에 부적합 ("0.35"→"0.36" 토큰이 완전히 다르지만 score 차이는 0.01)
4. **image_j 미사용**: 모델이 action 결과를 순수 "상상"으로 예측
5. **단일 scene**: Namaqualand만 학습
6. **target generation 문제**: 불가능한 occupancy+aspect 조합, c2o orthogonality vs 의미 충돌

## 8. 현재 진행 중: v2 학습

### 변경사항 (v1 → v2)
| | v1 | v2 |
|---|---|---|
| distance_threshold | 1.5m | **3.0m** |
| max_pairs_per_image | 32 | **64** |
| Pairs | ~17.6k | **~338k (19배)** |
| Epochs | 1 | **3** |
| Score keys | 13 | **14 (+visibility_ratio)** |
| Loss | LM only | **LM + Score Head regression (α=0.5)** |
| GPU | 1x H200 | **4x H200** |

### visibility_ratio 추가
- `src/scoring/evaluator.py`에 `TOP_LEVEL_SCORE_KEYS` 카테고리 추가
- annotation의 top-level `visibility_ratio` 필드를 14번째 score key로 사용
- 모델이 "이 action 후 사람이 얼마나 보일까"를 직접 예측 → occupancy 함정 방지

### Score Head (auxiliary regression loss)
- `model.score_head = Linear(2048 → 14)`
- prompt 끝 hidden state → 14개 float 직접 출력
- `total_loss = lm_loss + 0.5 * L1_regression_loss`
- `ScoreRegressionTrainer` 클래스 추가 (train.py)

### 상태
- Config: `configs/qwen35_vl_2b_4xh200_with_c2o_5k_v2.yaml`
- GPU 0~3 (4x H200), DeepSpeed ZeRO-3
- **예상 완료: ~7-9시간 (내일 오전)**
