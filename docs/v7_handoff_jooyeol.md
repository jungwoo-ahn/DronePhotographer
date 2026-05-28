# V7 Stage 2 + Stage 3 — 윤주열 핸드오프

내 쪽 (jungwooahn, 7×3090) Stage 2 풀 half-run **방금 시작**했음 (3,942 placements,
~8-10일 예상). 너도 같은 명령어로 본인 절반 (3,943 placements) 병렬로 시작하면 됨.
출력 디렉토리가 완전히 disjoint라 충돌 없음.

자세한 내용은 `docs/v7_stage2_3_pipeline.md` 참고. 아래는 빠른 실행 가이드.

---

## 0. 브랜치 pull

```bash
git fetch origin
git checkout v7_data_for_cosmos_policy
git pull --ff-only
```

새 파일들:
- `scripts/v7_stage2_render.py`            — Blender 렌더 + per-frame mesh bbox 계산
- `scripts/v7_stage2_backfill_bbox.py`     — 이미 렌더된 placement에 bbox만 후추가
- `scripts/v7_stage2_batch.sh`             — 멀티 GPU 런처
- `scripts/v7_stage3_score.py`             — V5 8-key CPU 스코어링
- `scripts/v7_stage2_make_split.py`        — split 매니페스트 생성기
- `splits/v7_stage2_assignments.json`      — 본인/내 placement 매니페스트 (commit됨)
- `docs/v7_stage2_3_pipeline.md`           — 상세 메모

## 1. 일회성 셋업 (네 머신 한 번만)

메인 DronePhotographer repo가 사이드 디렉토리에 있다면 (`../DronePhotographer/`)
batch script가 자동 감지. 아니면 심링크 직접 설정:

```bash
ROOT=$(pwd)  # /path/to/DronePhotographer-v7
MAIN=/path/to/DronePhotographer
ln -s "${MAIN}/blender"      "${ROOT}/blender"
ln -s "${MAIN}/data/scenes"  "${ROOT}/data/scenes"
ln -s "${MAIN}/data/objects" "${ROOT}/data/objects"
```

**GroundingDINO 설치 불필요.** torch CUDA 환경도 무관. Cycles는 Blender 내장
CUDA/OPTIX 사용, 스코어링은 순수 numpy.

## 2. Stage 2 풀 half-run (네 절반)

```bash
GPU_DEVICES="0 1 2 3 4 5 6 7" \        # 네 머신에서 가능한 GPU 인덱스
ASSIGNMENT_FILE=splits/v7_stage2_assignments.json \
SIDE=jooyeol \                          # ← 핵심: 너는 jooyeol
FRAMES_PER_PAIR=32 RENDER_SAMPLES=32 RESOLUTION="1024 768" \
RESUME=1 \
nohup bash scripts/v7_stage2_batch.sh > outputs/v7_stage2_batch_main.log 2>&1 &
```

- 슬라이스 로그: `outputs/v7_stage2_renders/_logs/slice_<i>.log`
- 메인 로그:    `outputs/v7_stage2_batch_main.log`
- 모니터링: `tail -F outputs/v7_stage2_renders/_logs/slice_0.log`
- 중단 후 재시작: `RESUME=1`이라 `done.flag` 있는 placement는 건너뜀

## 3. Stage 3 스코어링 (Stage 2 끝나고)

```bash
python3 scripts/v7_stage3_score.py \
    --out-dir outputs/v7_stage2_renders \
    --assignment-file splits/v7_stage2_assignments.json \
    --side jooyeol --resume
```

CPU only, 단일 프로세스. 본인 절반 전체 몇 분 안 걸림.

## 4. 산출물 (per placement)

```
outputs/v7_stage2_renders/<placement>/
├── renders/pair_<pp>_frame_<ff>.jpg   # K_accepted × 32 JPEGs
├── data.json                          # Stage 1 + render_records + scores
├── done.flag                          # Stage 2 완료
└── scored.flag                        # Stage 3 완료
```

`render_records[pair][frame]` 구조:
```json
{
  "frame_idx": 15,
  "path_rel": "renders/pair_00_frame_15.jpg",
  "bbox_xyxy_full": [x1, y1, x2, y2],
  "occupancy_clipped": 0.18,
  "in_frame": true,
  "scores": {
    "occupancy": 18, "body_in_frame_ratio": 95,
    "cam_to_obj_azimuth_deg": 243, "cam_to_obj_elevation_deg": -11,
    "object_center_x": 419, "object_center_y": 282,
    "bbox_x_offset": 117, "bbox_y_offset": 310
  }
}
```

8개 V5 키 — Qwen VLM 학습 타깃이랑 동일한 정수 스키마. v2 elev 컨벤션
(`cam 위 → elev<0`).

## 5. 실패 처리

스로우 발생한 placement는 `failed.flag` (Stage 2) 또는 `score_failed.flag`
(Stage 3) 가 traceback과 함께 생성됨.

```bash
find outputs/v7_stage2_renders -name '*failed.flag' | while read f; do
  echo "=== $(dirname $f) ==="
  head -20 $f
done
```

플래그 지우고 다시 launcher 돌리면 재시도.

## 6. 검증 (둘 다 끝나고)

```bash
# 겹침 없음
jq -r '.sides.jungwooahn.placements[]' splits/v7_stage2_assignments.json | sort > /tmp/jw.txt
jq -r '.sides.jooyeol.placements[]'    splits/v7_stage2_assignments.json | sort > /tmp/jy.txt
comm -12 /tmp/jw.txt /tmp/jy.txt   # 빈 줄이어야 함

# 커버리지 = Stage 1 전체
cat /tmp/jw.txt /tmp/jy.txt | sort -u | diff - <(ls outputs/v7_stage1_sample | sort)
```

---

## 진행 상황 (jungwooahn 측)

- ✅ Stage 1: 7,885 placements, 84,146 clips, ~2.69M frames (커밋 48148f60)
- ✅ Smoke 검증: 7 placement (다양한 scene) × 384 frames, OPTIX 3090 평균 ~4s/frame
- 🏃 **풀 half-run 진행 중** (방금 시작, ~8-10일 예상)
- ⏳ Stage 3 score (CPU, Stage 2 끝나고 자동 실행 가능)

질문 있으면 핑.
