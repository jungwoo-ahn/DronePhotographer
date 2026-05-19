# Iteration 1: v3 model (body_in_frame_ratio, 3 epochs) — 2026-04-19

## Model
- `runs/20260418_090117_qwen35_vl_2b_4xh200_with_c2o_5k_v2/final_converted`
- 5k, distance_threshold=3.0, max_pairs=64, 3 epochs, 14 keys (body_in_frame_ratio)
- Score Head regression loss (α=0.5)

## Inference Settings
- step=0.5m, rotation=8°, no lookahead, no adaptive step
- all weights=1.0, body_in_frame_ratio target=1.0
- seed=721, 50 steps

## Results

| Prompt | v1 (baseline) | v3-body (this) | Better? |
|--------|:---:|:---:|:---:|
| front eye-level | 0.176→0.068 (+61%) | 0.185→0.181 (+2%) | NO |
| left hero | 0.354→0.277 (+22%) | 0.304→0.296 (+3%) | NO |
| high angle thirds | 0.396→0.393 (+1%) | 0.390→0.363 (+7%) | YES |
| right cinematic | 0.380→0.281 (+26%) | 0.380→0.270 (+29%) | YES |

## 관찰

### 긍정적
- 사람이 프레임에 유지됨 — v1처럼 사람이 완전히 사라지는 스파이크가 없음
- body_in_frame_ratio 예측이 잘 동작 (평균 0.79~0.96, 적절한 범위)
- parse 성공률 99.6~99.9%
- right cinematic: v1보다 더 나은 결과 (0.270 vs 0.281)

### 문제점
1. **front eye-level이 거의 수렴 안함 (0.185→0.181, +2%)**
   - rollout summary: error가 0.10~0.27 사이를 왔다갔다 (진동)
   - best error=0.103이 step 중간에 달성되지만 유지 못함
   - 마지막 프레임: 사람 발 근처에서 꽃을 올려다보는 극단적 로우앵글
   - 카메라가 너무 낮게 내려감 (eye-level target인데)

2. **전체적으로 v1보다 수렴 속도가 느림**
   - v1은 13 keys, v3-body는 14 keys → 최적화 대상이 더 많아서 분산됨
   - body_in_frame_ratio weight=1.0이 다른 score들의 영향력을 희석

3. **카메라 drift 문제 여전함**
   - front eye-level 최종: 극단적 low angle (fz→양수 방향)
   - MPC greedy 특성이 해결 안 됨

## 다음 시도 방향
- body_in_frame_ratio weight를 줄이기 (1.0→0.5) — 다른 score 영향력 보존
- 또는 body_in_frame_ratio를 hard constraint로 사용 (target weight 아닌 필터)
- step size 줄이기 (0.5→0.3) — 진동 감소
- v1 모델 + body_in_frame 필터 조합 시도
