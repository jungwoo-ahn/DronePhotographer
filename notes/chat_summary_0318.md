# Chat Summary (2026-03-14 ~ 2026-03-18)
**Between: 안정우, 윤주열**

---

## 1. Training Progress
- H200 1장으로 학습 중, 약 하루 소요
- 현재 ~90% 학습 완료
- 60% 학습 시점에서 인퍼런스 테스트 진행 → VLM 기반이 기존 대비 훨씬 나음
- **총 학습 데이터**: 354,728 쌍 (train 347,634 / eval 7,094)
  - 기반 뷰: 10,000장
  - 거리 필터링(≤1.5m) 후 nearby pairs: 1.78M → target detection 적용 후 319K
  - zero-action pairs (10%): 35K 추가
  - 카메라 간 거리 2m~8m에서 촬영

## 2. Inference Results & Observations

### 테스트 구성
- **Case 1**: 가운데 위치 + subject size ~0.5 (가까이서 찍기)
- **Case 2**: 가운데 위치 + subject size ~0.3 + aspect ratio 1:1 
- **Case 3**: 오른쪽 위 삼분점 위치

### 결과
- 위 두 케이스: 가운데로 잘 감
- Case 1이 Case 2보다 subject 더 크게 (더 가까이서 찍은 느낌)
- Case 2 (1:1 aspect ratio): 위에서 찍어서 bbox가 정사각형 되길 기대했으나 효과 미미

### 한계점: 상반신 클로즈업 실패
- 시도 방법: bbox occupancy ratio 크게, 아래쪽 마진 없게, 좌/우/위 마진 있게, bbox aspect ratio 지정
- **실패 원인 1**: 학습 데이터가 2m~8m 거리에서만 촬영 → 근접 + 상반신샷 데이터 부족
- **실패 원인 2**: bbox 기준만으로는 모델이 "정면에서 예쁘게 상반신 자르기"를 할 이유 없음 — 정수리에서 적당히 찍은 것과 score 차이가 안 남
- **근본적 한계**: 3D 정보를 충분히 고려 못 함. "카메라 elevation 낮추고 가까이 다가가서 찍기" 같은 3D 추론 불가. 2D 정보를 외워서 "오른쪽 이동 → bbox 왼쪽 이동" 수준의 암기에 가까움

### 개선 방향 (인퍼런스)
- 뒷 스텝에서 더 조금씩만 움직이도록 제한 → 안정성 향상
- 3D bbox 관련 정보를 score로 활용 (orientation, depth 감각 등)

## 3. Photo Profile 기준 확장 논의 (윤주열 제안)
VLM으로 뽑는 score에 국한하지 말고, 우리가 원하는 기능 기준으로 생각해야 좋은 모델 나옴:

- **항공샷**: 찍게 할 것인지?
- **Close-up 퀄리티**: 물체 자르기(crop) 가능 여부, 상반신샷
- **피사체 지정**: output으로 피사체 text 나오게 해서 그걸 기준으로 움직이게?
- **Lighting**: 알아야 하는지?

## 4. Subject 지정 방식: Input vs Output
- **윤주열 제안**: 모델 output으로 피사체 text가 나오면 그걸 기준으로 움직이게 할 수 있지 않을까?
- **안정우 반론**: output이 아니라 input으로 넣어야 함. 다양한 scene에서 "누가봐도 subject인 애를 골라서 찍어라"를 학습하면, 찍을 만한 물체가 2개 이상일 때 모델이 헷갈림
- **현재 상태**: 씬 하나에서만 실험해서 subject text 없이 진행 중
- **합의**: VLM 전환 후 text input으로 넣어주는 게 어렵지 않음. GroundingDINO 돌릴 때 쓰던 프롬프트 그대로 활용 가능

## 5. Blender에서의 BBox 계산
- 윤주열: Blender에서 object를 scene에 넣으면 카메라 view 기준 bbox는 GroundingDINO 없이도 계산 가능
- 안정우: 맞음, 잘 계산하면 알 수 있음
- 다만 현재는 scene 안 물체를 찾아야 해서 GroundingDINO가 필요할 수도 있음

## 6. 데이터 수집 & 확장 계획
- 데이터 늘리기 필요 (윤주열이 알아보는 중)
- **근접 촬영 데이터**: 짧은 거리에서 찍은 데이터 다시 뽑아보기
- **피사체 scope**: 풍경 포함 여부 결정 필요
- **Scene 종류**: synthetic dataset 방 참고 → 종류 많음
- **이미지 샘플링 비율**: action 0 / action 작은 것 / action 큰 것 비중 정하기
- 안정우는 #1 (photo profile 기준) 부터 생각하기로 함

## 7. 학습 & 인퍼런스 최적화
- VRAM보다는 학습/인퍼런스 속도 개선에 집중
- toy 실험 1개에 1일 걸리는 것은 H200 치고 긴 편

## 9. 이번 주 미팅 목표
- 실험 스펙 보여주기: 시간, 메모리, 데이터
- 인퍼런스 방식 몇 가지 시연 (error 줄어드는 그래프 포함)
- 학습 디테일 추가 정리 (둘만 알고 정리)
