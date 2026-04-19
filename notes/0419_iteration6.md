# Iteration 6: Multi-seed generalization test — 2026-04-19

## Settings
- v1 model + margin filter + spike OFF + orig params (step=0.12m)
- Target: front eye-level (same)
- Seeds: 42, 123, 999 (4th seed=7777 didn't run due to for-loop issue)

## Results

| Seed | Error | Dist | Spikes | Status |
|:---:|:---:|:---:|:---:|------|
| 42 | - | - | - | HUNG at step 37 (killed) |
| 123 | **0.061** | 2.7→2.5m | 0 | Best ever! |
| 999 | 0.150 | 3.1→2.4m | 22 | Decent |
| ref: 721 | 0.068 | 4.7→2.3m | 15 | v1-orig baseline |

## 핵심 발견
- **seed=123 (0.061, 0 spikes)**: 역대 최고. 가까운 초기 위치 + margin filter 조합
- **seed=999 (0.150, 22 spikes)**: 수렴은 하지만 불안정
- **seed=42: step 37에서 hang** — vLLM inference hang 문제 재발. 특정 이미지에서 model.generate()가 멈추는 것으로 추정
- **초기 위치(seed)에 따라 결과 편차 큼** (0.061~0.150, 2.4x 차이)

## 결론
- 모델+config는 동작하지만, **초기 위치에 대한 robustness가 낮음**
- hang 문제는 vLLM 특유의 issue — timeout 메커니즘 필요
- margin filter + spike OFF + v1 orig params가 현재 가장 좋은 config
