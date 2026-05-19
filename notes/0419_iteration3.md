# Iteration 3: v1 model + safety guards + step=0.3m — 2026-04-19

## Settings
- Model: v1 (original, 13 keys, 1 epoch, threshold=1.5m)
- Safety guards ON (spike detection, margin filter)
- step=0.3m, no adaptive, no lookahead
- GPU3 failed (for loop race condition), only 3 experiments ran

## Results (3/4 targets)

| Prompt | v1-orig | it1(v3) | it2(v3) | it3(v1+guard) | Best |
|--------|:---:|:---:|:---:|:---:|:---:|
| front eye-level | **0.068** | 0.181 | 0.179 | 0.356 | v1 |
| left hero | 0.277 | 0.296 | **0.255** | 0.280 | it2 |
| high angle thirds | 0.393 | 0.363 | 0.360 | **0.315** | it3 |

## 핵심 관찰

### front eye-level 완전 실패
- error: 0.356, 50/50 steps 전부 spike (>0.2)
- 카메라가 위로 올라감 (z: 8.43→11.58, dist: 4.68→5.86m)
- safety guards의 spike detection이 v1 모델에서 오히려 방해:
  - v1은 c2o 예측이 불안정 → 매 step spike 감지 → 최소 이동 선택 → 전혀 진행 못함
  - v1-orig에는 spike detection 없어서 공격적으로 움직여 수렴

### high angle thirds 최고 결과
- 0.315: 모든 iteration 중 최저 error
- v1 모델이 이 target에서는 guards와 잘 맞음

### 결론
- **spike detection이 v1 모델의 적극적 이동을 막고 있음**
- v1-orig가 front에서 잘된 이유: spike에도 불구하고 공격적으로 접근 → 결국 수렴
- guards가 spike를 "안전하게" 처리하지만 결과적으로 움직임을 너무 억제

## 다음 시도
- v1 모델 + spike detection OFF + margin filter만 유지
- 또는 spike threshold를 더 높게 (0.5→0.8) 설정
