# Iteration 5: v1 model + orig params + margin filter + spike OFF — 2026-04-19

## Settings
- v1 model (13 keys, 1 epoch, threshold=1.5m)
- v1 original params: step=0.12m, rotation=3°, max_norm=0.18m
- v1 original weights: occ=2.0, centroid=2.0, rest=1.0
- margin filter ON, spike detection OFF, no adaptive, no lookahead
- All 4 targets ran successfully (individual launches, no for-loop)

## Results

| Prompt | v1-orig | it5 (v1+margin) | Dist | Spikes | Better? |
|--------|:---:|:---:|:---:|:---:|:---:|
| front eye-level | **0.068** | 0.108 | 4.7→4.0m | 1 | NO |
| left hero | 0.277 | 0.402 | 4.7→2.9m | 50 | NO |
| high angle thirds | 0.393 | **0.308** | 4.7→5.5m | 50 | YES |
| right cinematic | 0.281 | **0.195** | 4.7→5.4m | 47 | YES |

## 핵심 관찰

### front eye-level: 0.108 (spike 1개!)
- v1-orig (0.068)보다는 높지만, margin filter 덕에 spike가 15→1로 감소
- 최종 프레임: 사람 전신이 잘 보이는 좋은 구도 (아래에서 올려다봄)
- error 차이 (0.068 vs 0.108)는 주로 occupancy 미달 (카메라가 4.0m에서 멈춤)

### right cinematic: 역대 최고 (0.195!)
- v1-orig (0.281) 대비 30% 개선
- margin filter가 잘림 방지하면서 안정적 수렴

### left hero: 악화 (0.402)
- 50 steps 전부 spike — v1 모델의 c2o 불안정이 이 target에서 심함
- it4 (spike OFF, step=0.3m)에서 0.122였는데 step=0.12m으로 줄이니 악화

## BEST RESULTS ACROSS ALL ITERATIONS

| Prompt | Best Error | Config |
|--------|:---:|------|
| front eye-level | **0.068** | v1-orig (no guards, step=0.12m) |
| left hero | **0.122** | it4 (v1, spike OFF, step=0.3m) |
| high angle thirds | **0.308** | it5 (v1+margin, step=0.12m) |
| right cinematic | **0.195** | it5 (v1+margin, step=0.12m) |

## 결론
- **최적 config는 target마다 다름** — 하나의 세팅으로 모두 최적화 불가
- front: v1-orig의 공격적 접근이 최고
- left hero: 큰 step (0.3m) + spike OFF가 최고
- high/right: v1+margin (작은 step + margin filter)이 최고
- margin filter는 대체로 도움됨 (spike 감소, 안정성 증가)
- spike detection은 v1 모델에서 해로움 (움직임 억제)
