# render_object_v3 구현 요약 (2026-03-18)

## 목적

렌더링 시점에 Blender 3D bbox → 2D 투영 + 3D metrics 자동 계산.
GroundingDINO 없이 ground-truth bbox와 3D-aware score를 즉시 얻음.

## 생성 파일

| 파일 | 설명 |
|------|------|
| `render_object_v3.py` | `render_object.py`에서 utility import, 새 `main()` + 6개 함수 |
| `render_object_v3.sh` | `render_object.sh` 복사, script path / `RUN_NAME`만 변경 |

## 새 함수 (`render_object_v3.py`)

| 함수 | 설명 |
|------|------|
| `collect_object_meshes(name)` | object hierarchy 재귀 탐색 → MESH 리스트 수집 |
| `get_aabb_corners(min, max)` | AABB → 8 world vertices |
| `project_point_to_pixel(...)` | 3D point → pixel (Blender -Z forward convention) |
| `project_bbox_3d(...)` | 8 corners → 2D bbox + visibility stats |
| `compute_3d_metrics(...)` | elevation, azimuth, distance, visibility_ratio |
| `compute_object_dimensions(...)` | AABB → width / height / depth |

## object_name hierarchy 처리

`--object_name`에 parent object (`RIG-Snowman.001` 등) 주면
재귀적으로 모든 descendant MESH 수집 → combined AABB 계산.
AABB 중심 `(min+max)/2`를 카메라 타겟으로 사용 (`obj.location`보다 정확).

> 예: `RIG-Snowman.001` → 16개 MESH children → bbox 2.41 x 1.77 x 1.93m

## object_position 결정 우선순위

1. `--object_name` + `--object_position` : bbox는 object_name, 카메라 타겟은 object_position
2. `--object_name` only : AABB 중심을 카메라 타겟으로 사용
3. `--auto_place_object` : 3D bbox 불가, warning 출력, 3D fields 생략
4. `--object_position` only : 3D bbox 불가, warning 출력, 3D fields 생략

## Per-frame annotation 추가 필드

- `bbox_2d` — projected 2D bounding box `[x1, y1, x2, y2]`
- `projected_corners` — 8개 corner의 pixel 좌표
- `corners_in_front` / `corners_in_frame` — visibility 통계
- `elevation_deg` / `azimuth_deg` — 카메라-오브젝트 각도
- `camera_subject_distance` — 카메라-오브젝트 거리
- `visibility_ratio` / `truncation` — 가시성 정보
- `detections` — backward compat용 synthetic detection

## run_info.json 추가 섹션

```json
"object_3d": {
  "bbox_3d_corners": [[x,y,z], ...],
  "bbox_3d_min": [x, y, z],
  "bbox_3d_max": [x, y, z],
  "dimensions": {"width": ..., "height": ..., "depth": ...}
}
```

## Downstream 호환성

- v2 fields 전부 유지 → `dataset.py`, `evaluator.py` 그대로 동작
- `detections[].bbox_xyxy` → `bbox_control.py`, `score_annotations.py` 호환
- 새 3D fields는 기존 코드에서 무시됨

## 테스트 결과

DogWalk scene, `RIG-Snowman.001`, 5장 렌더 성공.
elevation 7~69°, azimuth 55~335°, distance 3.4~6.8m 정상 분포.

## Shell script 변경점 (`v3.sh`)

- `render_object.py` → `render_object_v3.py`
- `RUN_NAME` default: `v2_8k` → `v3_8k`
- `COMMON_ARGS`: `--object_position` 제거, `--object_name RIG-Snowman.001` 사용
