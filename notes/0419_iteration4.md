# Iteration 4: v1 model + spike OFF + margin ON + step=0.3m — 2026-04-19

## Settings
- v1 model (13 keys, 1 epoch, threshold=1.5m)
- spike detection OFF (--disable_spike_detection)
- margin filter ON
- step=0.3m, no adaptive, no lookahead
- Note: for-loop GPU issue → only 2/4 targets ran (front eye-level + left hero)

## Results

| Prompt | v1-orig (step=0.2) | it4 (v1+noSpike, step=0.3) |
|--------|:---:|:---:|
| front eye-level | **0.068** | 0.323 (50 spikes) |
| left hero | 0.277 | **0.122** (0 spikes) |

## 핵심 발견

### left hero: 역대 최고 결과 (0.122)
- v1 모델 + spike OFF + margin filter = 최적 조합
- 카메라가 적극 접근 (4.7→2.5m), spike 없이 안정적
- 사람 상체가 크게 잘 보이는 좋은 구도

### front eye-level: 여전히 나쁨 (0.323)
- 50/50 steps 전부 error>0.2
- step=0.3m인데 v1-orig는 step=0.2m
- margin filter가 front 접근을 방해할 수 있음 (사람이 프레임 상단에 닿으면 필터)

## 종합 분석 (4 iterations)

| Config | front | left | high | right | 특징 |
|--------|:---:|:---:|:---:|:---:|------|
| v1-orig (step=0.2, no guard) | **0.068** | 0.277 | 0.393 | 0.281 | 접근 잘함, spike 있음 |
| it1 (v3, w=1, step=0.5) | 0.181 | 0.296 | 0.363 | **0.270** | 안정적이지만 느림 |
| it2 (v3, w=.5, step=.3) | 0.179 | **0.255** | 0.360 | 0.373 | drift 적음 |
| it3 (v1+guard, step=.3) | 0.356 | 0.280 | **0.315** | N/A | spike detection 방해 |
| it4 (v1+noSpike, step=.3) | 0.323 | **0.122** | N/A | N/A | left hero 역대 최고 |

## 다음 시도
- v1-orig 세팅 그대로 (step=0.2, no guards) + margin filter만 추가
  → v1의 성공 비결을 유지하면서 잘림만 방지
