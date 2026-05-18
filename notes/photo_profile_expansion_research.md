# Photo Profile Criteria Expansion + Research Ideas

## 배경

DronePhotographer는 VLM-based drone camera composition optimizer로 동작함: Blender가 synthetic view를 render하고 -> GroundingDINO가 subject를 detect하고 -> VLM(Qwen)이 (image + action)에서 composition score를 예측하도록 학습하고 -> MPC가 target composition 방향으로 camera movement를 planning함.

**Core problem**: 현재 scoring system은 7개의 rule-based 2D bbox metric(occupancy, margins, aspect ratio, centroid offset)만 사용함. 모델이 2D bbox pattern은 잘 memorize하지만 3D spatial relationship은 reasoning하지 못해서, "move camera right -> bbox shifts left" 수준에 머물고 "lower elevation + approach -> upper-body close-up" 수준 제어는 어려움.

**Goal**: photo profile criteria를 확장해서 pitch/elevation, close-up, aerial, depth 같은 richer control을 가능하게 하고, 최근 paper(FeedbackDescent, VLA, spatial VLM 등) 기반으로 시스템 개선 research direction을 정리함.

---

## Part 1: Photo Profile Criteria Expansion

### 1A. New 3D-Aware Score Keys to Add

아래 key들은 기존 annotation data(`camera_position`, `object_position`, `final_forward`, `final_up`)만으로 계산 가능해서, 추가 rendering 없이 바로 도입 가능함.

| New Score Key | Definition | Range | Why |
|---|---|---|---|
| `camera_elevation_angle` | arcsin((cam_z - obj_z) / distance) | [-1, 1] normalized | Enables aerial vs eye-level vs low-angle control |
| `camera_azimuth_angle` | atan2(cam_y - obj_y, cam_x - obj_x) | [0, 1] normalized | Controls shooting direction relative to subject |
| `camera_subject_distance` | Euclidean distance, normalized by max radius | [0, 1] | Close-up vs wide shot control |
| `camera_pitch_angle` | angle between forward vector and horizontal plane | [-1, 1] normalized | Looking down (aerial) vs up (heroic) vs level |

**수정할 파일:**
- `src/scoring/bbox_control.py` — 기존 `compute_rule_based_scores()` 옆에 `compute_3d_aware_scores()` 추가하기
- `src/scoring/evaluator.py` — `ALL_SUPPORTED_SCORE_KEYS`에 new key 등록하기
- `src/vlm_qwen25/schema.py` — `SCORE_KEYS`에 new key 추가하기
- `src/vlm_qwen25/dataset.py` — `ViewRecord`에 3D info(object_position, radius) 전달하고 3D score 계산하기
- `src/vlm_qwen25/objective.py` — new preset(`aerial_centered`, `low_angle_heroic`, `eye_level_portrait` 등) 추가하기
- Config YAML — `target_score_keys`에 new key 추가하기

### 1B. New Presets Enabled by 3D Scores

```python
# examples
"aerial_centered": {
    "center_x": 0.5, "center_y": 0.5,
    "occupancy": 0.4, "aspect_ratio": 1.0,
    "camera_elevation_angle": 0.8,  # high above
    "camera_pitch_angle": -0.7,     # looking down
}
"eye_level_portrait_close": {
    "center_x": 0.5, "center_y": 0.45,
    "occupancy": 0.5, "aspect_ratio": 0.67,
    "camera_elevation_angle": 0.0,   # eye level
    "camera_subject_distance": 0.2,  # close
}
"heroic_low_angle": {
    "center_x": 0.5, "center_y": 0.6,
    "occupancy": 0.45, "aspect_ratio": 1.0,
    "camera_elevation_angle": -0.3,  # below eye level
    "camera_pitch_angle": 0.3,       # looking up
}
```

### 1C. Close-Up Data Gap

현재 rendering은 `camera_radius_range 2 8`(2m-8m)만 사용함. Close-up이 실패하는 이유는 다음과 같음:
- 0.5-2m range training data가 없음
- close range에서 GroundingDINO bbox가 heavily cropped될 수 있음

**Fix**: `--camera_radius_range 0.5 2`로 re-render해서 close-range data를 만들고, 기존 dataset과 merge하면 됨. 추가로 cropped subject에서도 bbox metric은 여전히 유효함(`margin_bottom = 0`이면 subject가 bottom edge까지 내려온 상태), 그래서 upper-body framing에는 오히려 유용할 수 있음.

### 1D. Depth Stratification & Ground Plane (lower priority)

- `--render_depth`로 Depth map은 이미 render 가능하지만 아직 사용하지 않음
- non-subject region의 depth variance로 `background_complexity` score 계산을 고려할 수 있음
- Ground plane placement는 3D object position + camera transform으로 계산 가능하며, object base가 frame 어디에 projection되는지 알 수 있음
- 다만 위 항목들은 1A 대비 구현 복잡도가 높고 immediate impact는 낮음

---

## Part 2: FeedbackDescent Application

### Paper Summary
FeedbackDescent(Stanford, 2025)는 pairwise comparison + textual rationale로 text artifact를 최적화함. feedback을 scalar reward로 압축하지 않고 structured text feedback을 "gradient"처럼 유지하며, LLM in-context learning으로 출력을 iterative하게 개선함. Inference-time 방식이고 weight update는 없음.

### How to Apply to DronePhotographer

**Idea: FeedbackDescent for MPC action selection refinement**

현재 MPC는 discrete candidate action을 만들고 -> 각 action을 VLM으로 score하고 -> lowest error를 고르는 brute-force grid search 구조임.

FeedbackDescent 대안:
1. VLM이 current frame + proposed action을 보고 score를 예측함
2. score만 받지 않고 **textual critique**도 함께 출력하게 함: "subject가 너무 작고 camera angle이 너무 높다. forward로 이동하고 elevation을 낮추면 개선된다."
3. 이 textual feedback으로 grid 전체를 다 도는 대신 **next action candidate를 refine**함
4. propose -> critique -> refine -> critique -> select 순서로 반복함

**Benefits:**
- step당 VLM forward pass 수를 줄일 수 있음(critique-guided search vs brute-force grid)
- scalar score만 쓸 때보다 directional information이 더 풍부해짐
- spatial planning에 VLM의 language reasoning capability를 활용할 수 있음
- 3D action을 natural language로 reasoning해서 "2D memorization" 한계를 완화할 수 있음

**Implementation approach:**
- `src/vlm_qwen25/mpc.py`를 확장해서 기존 grid search와 함께 `feedback_descent` mode를 지원하게 함
- score + textual rationale를 요청하는 new prompt template을 추가함
- iterative refinement loop를 구현함: initial action -> VLM critique -> adjusted action -> re-score -> select

**Risk:** 유의미한 spatial reasoning을 VLM이 생성할 수 있어야 효과가 남. CoT 가능한 Qwen 또는 더 큰 model이 필요할 수 있음.

---

## Part 3: Other Research-Inspired Ideas

### 3A. History / Sequence Input (High impact, moderate effort)

**Inspiration**: VLM2 (memory for spatial reasoning), NaVid (video-based VLN), UP-VLA (sequence prediction)

**Current limitation**: VLM이 한 번에 frame 하나만 보고 temporal context가 없음.

**Proposal**: 마지막 N frame(예: 3-5)을 multi-image input으로 VLM에 넣음. 이렇게 하면:
- motion trajectory context(어디로 이동해왔는지)를 제공할 수 있음
- multiple viewpoint 기반 implicit 3D scene understanding이 가능해짐
- action consistency가 좋아져 oscillation을 줄일 수 있음

**Implementation**: Qwen2.5-VL이 multi-image input을 이미 지원하므로, `collator.py`, `dataset.py`를 수정해서 history frame을 포함하면 됨. MPC inference 시에는 frame buffer를 유지하면 됨.

### 3B. VLMPC-Style Learned Dynamics (High impact, high effort)

**Inspiration**: VLMPC (RSS 2024) — almost identical architecture to DronePhotographer

**Key difference**: VLMPC는 action을 넣으면 *future visual observation*을 예측하는 learned dynamics model을 사용하고, 그 예측 결과를 score함.

**Application**: Blender re-render(slow)나 pre-rendered view snap(approximate) 대신 lightweight video prediction model을 학습해서 "이 action 뒤 scene이 어떻게 보일지"를 상상하고, 그 imagined view를 score함. 이렇게 하면 rendering bottleneck을 줄일 수 있음.

### 3C. SpatialVLM-Style 3D Reasoning (High impact, moderate effort)

**Inspiration**: SpatialVLM (CVPR 2024) — endows VLMs with metric spatial reasoning

**Application**: composition prediction과 함께 spatial QA를 같이 fine-tune함. 예를 들어 아래 같은 training sample을 추가함:
- "How far is the camera from the subject?" → "2.3m"
- "What is the camera elevation relative to the subject?" → "15 degrees above"

이 접근은 VLM이 2D bbox pattern만 memorize하지 않고 internal 3D understanding을 학습하는 데 도움이 될 수 있음.

### 3D. Test-Time Reasoning / Chain-of-Thought (Medium impact, low effort)

**Inspiration**: o1, DeepSeek-R1, Forest-of-Thought

**Application**: MPC inference 중에 score를 출력하기 전에 VLM에게 "think step by step" 하도록 유도함:
```
"First, I notice the subject is in the upper right. The proposed action moves the camera left and forward.
This should shift the subject more toward center and increase its size.
Predicted scores: {...}"
```

inference당 token cost가 늘어나는 대신 score prediction accuracy가 좋아질 수 있음. reasoning이 더 강한 Qwen3.5로 테스트할 가치가 있음.

### 3E. RL Fine-Tuning with Composition Reward (Medium impact, high effort)

**Inspiration**: Human-in-the-loop RL (Science Robotics 2024), GRPO

**Application**: supervised VLM training 이후 RL fine-tuning을 수행함:
- Reward = actual composition improvement (measured by Blender re-render + GroundingDINO)
- Policy = VLM predicting action given image
- 이렇게 하면 "given action의 score 예측"과 "좋은 action 선택" 사이의 gap을 줄일 수 있음

### 3F. Graph-Based Composition Understanding (Low-medium impact)

**Inspiration**: Graph-based aesthetic assessment papers

**Application**: bbox score에 graph-structured composition feature(여러 detected object 간 spatial relationship, leading line, symmetry)를 augment함. 다만 annotation pipeline 복잡도가 올라감.

---

## Priority Ranking

| Priority | Item | Impact | Effort | Dependency |
|---|---|---|---|---|
| **P0** | 1A: Add 3D-aware scores (elevation, distance, pitch) | High | Low | None — computable from existing annotations |
| **P0** | 1C: Re-render close-range data (0.5-2m) | High | Low | Just re-run render_object.sh with new radius |
| **P1** | 3A: History/sequence input | High | Medium | Need to modify dataset + collator |
| **P1** | 1B: New composition presets | Medium | Low | Depends on 1A |
| **P2** | 3D: Test-time CoT reasoning | Medium | Low | Just prompt engineering |
| **P2** | FeedbackDescent for MPC | Medium | Medium | Need new MPC mode |
| **P3** | 3C: SpatialVLM-style spatial QA | High | Medium | Need additional training data |
| **P3** | 3B: Learned dynamics model | High | High | Separate model training |
| **P4** | 3E: RL fine-tuning | Medium | High | Need reward infrastructure |
| **P4** | 1D: Depth stratification | Low-Med | Medium | Need depth rendering pipeline |

---

## Key Papers Reference

| Paper | Venue | Core Relevance |
|---|---|---|
| **VLMPC** (RSS 2024) | RSS | Same architecture — VLM + MPC for control |
| **SpatialVLM** (CVPR 2024) | CVPR | 3D spatial reasoning in VLMs |
| **FeedbackDescent** (Stanford 2025) | ArXiv | Inference-time optimization via textual gradients |
| **UP-VLA** (2025) | ICLR-adj | Unified understanding + action prediction |
| **VLM2** (2025) | — | Memory for spatial reasoning across frames |
| **DroneVLA** (2025) | ArXiv | VLA for aerial manipulation |
| **Learning Camera Movement from Drone Videos** (2024) | ArXiv | Learning camera control from real footage |
| **Forest-of-Thought** (2024) | ArXiv | Test-time compute for reasoning |

---

## Verification

P0 항목 구현 후 아래 순서로 검증함:
1. 3D score computation 추가 -> `scripts/score_annotations.py` 실행 -> annotation에 new key가 생겼는지 확인함
2. training config의 `target_score_keys` 업데이트 -> small model 학습 -> loss convergence 확인함
3. `objective.py`에 new preset 추가 -> `scripts/infer_mpc.py --target_preset aerial_centered` 실행 -> MPC가 elevation control을 목표로 동작하는지 확인함
4. close-range data re-render -> annotation merge -> 재학습 -> close-up preset 성능 확인함
