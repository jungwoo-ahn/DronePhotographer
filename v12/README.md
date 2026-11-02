# V12 snapshot: the 3DGS benchmark (paper Table 5) and the investigation

A copy of [DronePhotographerV12 @ b6d9090](https://github.com/jungwoo-ahn/DronePhotographerV12/commit/b6d9090)
(the files of that commit, same relative paths), placed here so it can be read next to the rest
of the project. Context and results in plain words: V12 issue
[#4](https://github.com/jungwoo-ahn/DronePhotographerV12/issues/4) and the matching issue in this repo.

- **Benchmark:** `configs/eval_episodes_gs_v1.json` (90 episodes, 5 reconstructed scenes x 10
  subjects); built by `scripts/gs_build_episodes.py` + `scripts/gs_select_episodes.py`; rendered by
  `scripts/gs_render_server.py` + `scripts/gs_subject_blender.py` (`src/common/gs_render.py` is the
  rollout client); run with `scripts/submit_bench_gs.sh`; scored by `scripts/score_benchmark.py`.
- **Investigation:** `scripts/gs_investigation/README.md` (why every shot-profile policy scores
  0-2% here), `scripts/build_spin_episodes.py`.
- The splat/subject compositing helpers these import live in this repo at
  `data/gs_scenes/_tools/composite/` (`gscommon.py` etc., committed alongside).

The scripts use absolute NAS paths (`/home/nas_main/jungwooahn/projects/DronePhotographerV12`,
`.../DronePhotographer`) and the V12 venvs; run them from the V12 checkout. All results are on NAS
under `DronePhotographerV12/runs/` (`bench_gs`, `bench_gs_probe`, `bench_v2`, `bench_v2_spin180`).
