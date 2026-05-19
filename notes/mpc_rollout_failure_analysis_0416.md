# MPC Rollout Failure Analysis (2026-04-16)

## 실험 조건
- Model: `runs/20260403_151944_qwen35_vl_2b_1xh200_with_c2o_5k/final` (Qwen3.5-2B, c2o 포함)
- Scene: `outputs/Namaqualand_namaqualand_v3_260401_024633`
- Inference: vLLM 0.19.0, Blender-in-the-loop, 50 steps
- 4개 실험 (front eye-level / left hero / high angle thirds / right cinematic)

## 결과 요약

| Prompt | Initial Error | Final Error | 감소율 | 비고 |
|--------|:---:|:---:|:---:|------|
| front eye-level, centered, medium | 0.1762 | 0.0680 | 61.4% | 중간에 에러 스파이크 반복 |
| left profile, hero, close-up | 0.3540 | 0.2765 | 21.9% | 수렴 느림 |
| high angle, thirds, wide | 0.3958 | 0.3928 | 0.8% | 거의 수렴 안 됨 |
| right side, cinematic wide | 0.3796 | 0.2815 | 25.9% | 느린 수렴 |

## 핵심 문제: 사람이 프레임에서 사라지면 모델이 무너짐

### 현상 (Exp1 기준)
- Step 0~16: error 정상 감소 (0.18 → 0.08), 모델이 fy≈+1.0 예측
- Step 17: 갑자기 error 0.30으로 스파이크, 모델이 fy=-0.615 예측
- Step 19~28: fy=-0.431 동일값 반복 출력, error 0.27~0.29 유지
- Step 29: 우연히 사람이 다시 보이면서 복구 (error 0.08)
- Step 34~38: 또 같은 패턴 반복

### 프레임 확인
- Step 10 (error=0.07): 사람 상체~하체 잘 보임
- Step 16 (error=0.09): 발만 프레임 상단에 보임 (카메라 너무 낮고 가까움)
- Step 17 (error=0.30): 사람 완전히 사라짐, 땅바닥만 보임

## 원인 분석

### 1. 사람 가시성 확인 없음 (시스템)
- MPC가 모델 예측만 사용, 사람이 실제로 프레임에 있는지 확인하지 않음
- `--evaluate_with_detector` 미사용 → GroundingDINO 실제 bbox 확인 없음
- 모델 예측을 100% 신뢰하므로, 잘못된 예측 → 잘못된 액션 → 더 나빠지는 악순환

### 2. 카메라 너무 가까이 접근 (타겟/시스템)
- occupancy=0.4 타겟 → 카메라가 계속 가까이 끌려감
- 거리: 4.68m (step 0) → 2.03m (step 40)
- 가까울수록 작은 이동에도 시야가 크게 변함

### 3. c2o 예측 불안정 (모델)
- 사람이 조금만 잘려도 c2o 예측이 급격히 나빠짐
- 사람이 안 보이면 fy=-0.431 같은 특정값 반복 출력 (mode collapse)
- bbox 예측은 상대적으로 안정적, c2o 예측이 주 문제

### 4. 스텝 크기 대비 거리 (시스템)
- 2~3m 거리에서 0.2m 스텝 = 한 번에 ~6° 시야 변화
- 경계 조건 근처에서 유효 candidate 수 급감 (720 → 388, 348)

### 5. detector 피드백 없음 (시스템)
- 현재: 모델 predicted scores만 사용
- actual scores (GroundingDINO 기반)를 MPC loop에 통합하면 개선 가능

## 개선 방향 (TODO)

1. **사람 가시성 체크 추가**: 매 스텝 GroundingDINO로 사람 detection, 안 보이면 이전 위치로 롤백 또는 후퇴 액션
2. **adaptive step size**: 거리에 비례해서 이동량 조절 (가까우면 작게)
3. **occupancy 상한 클램핑**: 카메라가 너무 가까이 가지 않도록 최소 거리 제한
4. **c2o prediction confidence**: 모델 출력 신뢰도 낮으면 해당 candidate 패널티
5. **detector 기반 actual score feedback**: predicted가 아닌 actual bbox 기반 error를 MPC에 반영
6. **모델 학습 개선**: c2o 예측 데이터 보강, 사람이 부분적으로 보이는 케이스 학습

## 참고: 타겟 생성은 문제 아님
- generate_target.py (claude-cli backend) 가 생성한 c2o 벡터는 합리적
- 정면(fy=1.0), 왼쪽(fx=0.7), 위에서(fz=-0.7) 등 의도대로 생성됨
- 문제는 타겟 자체가 아니라 MPC 실행 과정에서 발생
