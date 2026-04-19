# Iteration 2: body_in_frame weight=0.5, step=0.3m — 2026-04-19

## Settings
- Same model as it1 (v3-body, 3 epochs)
- Changed: body_in_frame_ratio weight 1.0→0.5, step 0.5m→0.3m

## Results

| Prompt | v1 | it1 (w=1,s=.5) | it2 (w=.5,s=.3) | Best |
|--------|:---:|:---:|:---:|:---:|
| front eye-level | **0.068** | 0.181 | 0.179 | v1 |
| left hero | 0.277 | 0.296 | **0.255** | it2 |
| high angle thirds | 0.393 | 0.363 | **0.360** | it2 |
| right cinematic | 0.281 | **0.270** | 0.373 | it1 |

## 관찰

### step=0.3m이 camera drift 줄임
- front eye-level z변화: v1=-1.60m, it1=-1.64m, it2=-0.33m
- 작은 step이 drift 방지에 효과적
- 하지만 접근도 못함 (dist 4.68→5.94m, 오히려 멀어짐)

### 핵심 문제: v3 모델이 "다가가는 법"을 모름
- v1은 dist 4.68→2.26m (적극 접근)
- v3(it1,it2)는 dist 4.68→5.94m (오히려 멀어짐!)
- 모델이 바뀌면서 접근 방향 예측 능력이 약해짐
- body_in_frame_ratio가 "프레임에 유지"는 시키지만 "다가가기"를 방해할 수도

### left hero에서 it2가 v1보다 나음 (0.255 < 0.277)
- 작은 step + 낮은 body weight 조합이 어려운 target에서 효과적

### right cinematic에서 it2 악화 (0.373 vs 0.270)
- step=0.3m이 너무 작아서 큰 이동 필요한 target에 불리

## 핵심 인사이트
- **v1 모델은 접근을 잘하지만 사람을 놓침** (spike 문제)
- **v3 모델은 사람을 유지하지만 접근을 못함** (conservative)
- 이상적: v1의 공격적 접근 + v3의 안정성 결합

## 다음 시도
- v1 모델 + body_in_frame_ratio 필터 (학습 없이 inference에서만 체크)
  → v1의 예측력 유지 + body_in_frame으로 잘림만 방지
- 또는 body_in_frame_ratio weight=0으로 (target에서 제외) + margin filter만 사용
