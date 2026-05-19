# Iteration 9: Per-target optimal configs × 100 steps — 2026-04-19

## Settings (target-specific)
| GPU | Target | Config |
|-----|--------|--------|
| 0 | front eye-level, seed=721 | v1-orig (spike allowed, no margin) |
| 1 | left hero, seed=721 | step=0.3m + spike OFF |
| 2 | high angle thirds, seed=721 | margin + spike OFF |
| 3 | front eye-level, seed=123 | spike OFF |

## Results

| Config | @50 | @100 | Prev Best | New? |
|--------|:---:|:---:|:---:|:---:|
| front v1-orig | 0.115 | 0.117 | **0.061** | no |
| left step=0.3 | 0.237 | 0.206 | **0.122** | no |
| high spikeOFF | 0.341 | 0.326 | **0.308** | no |
| front s=123 | 0.093 | 0.143 | **0.061** | no |

## 핵심 관찰
- **100 steps에서도 이전 best를 못 이김** — 이번 결과들이 이전보다 나쁜 이유:
  - front v1-orig: 0.117 vs it8의 같은 config 0.282 → 비슷한 수준. 하지만 v1-orig 50 steps (0.068)에 못 미침
  - left step=0.3: 0.206 vs it4의 0.122 → for-loop race condition 문제인가? it4는 단독 실행이었음
  - front s=123: step 50에서 0.093 (좋음!) → step 100에서 0.143으로 악화. 오버슈팅.

- **front s=123 오버슈팅 문제**: 50 steps에서 0.093이었는데 100 steps에서 0.143으로 악화. 수렴 후 발산. MPC greedy의 한계 — 최적점을 지나치면 되돌아오지 못함.

## 결론
- 100 steps가 항상 도움되는 건 아님 — 일부 target에서는 오버슈팅으로 악화
- 최적 step 수도 target/seed에 따라 다름
- 이전 best들 (it4, it5, it6, it8)이 여전히 최고

## 루프 중단 사유
- 9 iterations 동안 충분한 탐색 완료
- diminishing returns: 새로운 config 변경이 이전 best를 넘지 못함
- 근본적 개선은 모델 학습 또는 MPC 알고리즘 변경 필요
