# Iteration 8: 100 steps convergence test — 2026-04-19

## Settings
- v1 model + margin filter + spike OFF + orig params (step=0.12m)
- Seed: 721, **100 steps** (2x normal)
- 4 targets: front, left hero, high angle thirds, right cinematic

## Results: 50 steps vs 100 steps

| Target | v1@50 | it8@50 | it8@100 | Improved? | Spikes |
|--------|:---:|:---:|:---:|:---:|:---:|
| front eye-level | 0.068 | 0.284 | 0.282 | barely | 100/100 |
| left hero | 0.277 | 0.294 | **0.178** | yes | 86/100 |
| high angle thirds | 0.393 | 0.440 | **0.320** | yes | 100/100 |
| right cinematic | 0.281 | 0.110 | **0.067** | best ever! | 8/100 |

## 핵심 발견

### right cinematic: 0.067 (역대 최고!)
- 50 steps에서 0.110이었는데 100 steps에서 0.067으로 크게 개선
- 카메라가 4.7→3.4m로 접근, spike 8개만 (8%)
- 더 많은 steps가 수렴을 크게 도움

### left hero, high angle: 100 steps에서 유의미한 개선
- left hero: 0.294→0.178 (39% 개선)
- high angle: 0.440→0.320 (27% 개선)
- 하지만 여전히 spike 많음 (86%, 100%)

### front eye-level: 100 steps에도 수렴 안 됨
- 0.284→0.282 (거의 변화 없음)
- 100 steps 전부 spike — 이 config에서 이 target은 근본적으로 안 됨

## 결론
- **100 steps가 도움됨**: right cinematic, left hero, high angle 모두 개선
- **right cinematic 0.067**: 8 iterations 전체 최고 결과
- **front eye-level은 이 config (v1+spikeOFF+margin)에서 불가** — v1-orig (spike 허용)에서만 0.068 달성
- **수렴이 느린 target일수록 steps 증가 효과 큼**

## 전체 BEST RESULTS 업데이트

| Target | Best Error | Steps | Config | Seed |
|--------|:---:|:---:|------|:---:|
| front eye-level | **0.061** | 50 | v1+margin+spikeOFF | 123 |
| left hero | **0.122** | 50 | v1+spikeOFF+step=0.3 | 721 |
| high angle thirds | **0.308** | 50 | v1+margin | 721 |
| right cinematic | **0.067** | 100 | v1+margin+spikeOFF | 721 |
