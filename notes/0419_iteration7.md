# Iteration 7: seed=123 × 4 different targets — 2026-04-19

## Settings
- v1 model + margin filter + spike OFF + orig params (step=0.12m)
- Seed: 123 (best from it6 front eye-level test)
- 4 targets: left hero, high angle thirds, right cinematic, three-quarter above

## Results

| Target | Prev Best (seed=721) | It7 (seed=123) | Spikes | New Best? |
|--------|:---:|:---:|:---:|:---:|
| left hero | **0.122** (it4) | 0.271 | 50 | no |
| high angle thirds | **0.308** (it5) | 0.356 | 50 | no |
| right cinematic | **0.195** (it5) | 0.206 | 37 | no |
| three-quarter above | N/A | **0.168** | 1 | new baseline |

## 핵심 발견
- **seed=123은 front에서만 좋고, 다른 target에서는 seed=721보다 나쁨**
- seed=123의 초기 위치(dist=2.7m)가 이미 가까워서 front에 유리하지만, 큰 각도 변환이 필요한 target에는 불리
- **three-quarter above: 0.168, spike 1개** — 새 target이지만 양호한 결과
- left hero와 high angle: 50 spikes — seed=123 위치에서 이 target으로의 경로가 어려움

## 종합 결론 (7 iterations)

### 최적 결과 종합
| Target | Best Error | Config | Seed |
|--------|:---:|------|:---:|
| front eye-level | **0.061** | v1+margin+spikeOFF | 123 |
| left hero | **0.122** | v1+spikeOFF+step=0.3 | 721 |
| high angle thirds | **0.308** | v1+margin | 721 |
| right cinematic | **0.195** | v1+margin | 721 |
| three-quarter above | **0.168** | v1+margin+spikeOFF | 123 |

### 핵심 인사이트
1. **v1 모델이 v3(body_in_frame) 모델보다 일관되게 나음** — 추가 학습이 오히려 해로웠을 수 있음
2. **최적 seed는 target마다 다름** — 초기 위치와 목표 포즈의 궁합이 성능을 좌우
3. **margin filter는 대체로 도움됨** (spike 감소)
4. **spike detection은 v1 모델에서 해로움** (움직임 억제)
5. **step size 0.12m (v1 orig)이 대부분의 target에 최적** — 단 left hero는 0.3m이 더 나음
