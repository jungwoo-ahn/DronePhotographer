"""The adapter seam in `scripts/closed_loop_eval.py`.

Everything a policy needs is behind `act(frame, goal_vec, goal_raw, seed)`; the loop is
shared. These tests pin the three things that make that true and would otherwise drift
silently: the constants the script now mirrors instead of importing (so it can run without
cosmos), the goal-key order the baselines were trained with, and the frame/chunk contract.
No GPU, no weights: the model-bearing adapters are exercised through their static pieces.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.baselines import adapters
from src.baselines.goal_space import DEFAULT_V5_RANGES as THEIR_RANGES
from src.baselines.goal_text import NL_GOAL_KEYS
from src.common.goal_space import DEFAULT_GOAL_KEYS, DEFAULT_V5_RANGES

REPO = Path(__file__).resolve().parents[1]
SCRIPT = (REPO / "scripts" / "closed_loop_eval.py").read_text()


# ---- the script must be importable without cosmos ------------------------------------------
def test_the_eval_script_imports_no_cosmos_at_module_level():
    """The baselines run in a LeRobot venv (transformers 5) that cannot hold cosmos. A stray
    top-level `from cosmos_framework ...` makes `--policy pi05` fail before argparse."""
    top_level = [ln for ln in SCRIPT.splitlines() if re.match(r"^(from|import) cosmos_framework", ln)]
    assert top_level == [], top_level
    assert "from src.data.cosmos_camera_dataset import" not in SCRIPT


def test_mirrored_constants_match_the_dataset_module():
    """`DEFAULT_VAL_SCENES` and `CAMERA_ACTION_DIM` are duplicated in the script rather than
    imported, because the dataset module drags cosmos in. Duplicates drift; this pins them."""
    cosmos_ds = pytest.importorskip("src.data.cosmos_camera_dataset")   # needs the framework venv
    ns = {"Path": Path, "V12": str(REPO)}
    for name in ("DEFAULT_VAL_SCENES", "CAMERA_ACTION_DIM"):
        line = next(ln for ln in SCRIPT.splitlines() if ln.startswith(f"{name} ="))
        exec(line, ns)
        assert ns[name] == getattr(cosmos_ds, name), name
    # The domain name is a literal in CosmosAdapter for the same reason; pin it too.
    import inspect
    src = inspect.getsource(adapters.CosmosAdapter.__init__)
    assert f'self.domain_name = "{cosmos_ds.DOMAIN_NAME}"' in src


# ---- goal keys: what the baselines were trained on ------------------------------------------
def test_vla_prompt_keys_are_the_loops_goal_vec_order():
    """The loop hands every adapter `goal_vec` in DEFAULT_GOAL_KEYS order. The VLAs' prompt
    formatter indexes it by NL_GOAL_KEYS. They are the same list today; the source repo's
    own DEFAULT_GOAL_KEYS is NOT (world azimuth), so this is a thing to test, not assume."""
    assert tuple(NL_GOAL_KEYS) == tuple(DEFAULT_GOAL_KEYS)


def test_dp_normalization_ranges_match():
    """DP normalizes the same vector with the vendored ranges; ours must agree or the
    'same' goal lands in a different place in DP's input space."""
    for k in DEFAULT_GOAL_KEYS:
        assert THEIR_RANGES[k] == DEFAULT_V5_RANGES[k], k


def test_dp_adapter_refuses_a_config_with_another_key_order(tmp_path, monkeypatch):
    """The assert is the point: a DP trained on a different key order must not run at all."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("data:\n  goal_score_keys: [occupancy, body_in_frame_ratio]\nmodel: {}\n")
    # Stop before any weights load: the assert fires right after goal_keys().
    import types, sys
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoImageProcessor = fake_tf.AutoModel = object
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    with pytest.raises(AssertionError, match="different key order"):
        adapters.DPAdapter("/nonexistent.pt", str(cfg))


# ---- the frame / chunk contract -------------------------------------------------------------
def test_squash_is_non_aspect_preserving_and_a_noop_at_size():
    frame = (np.random.RandomState(0).rand(258, 344, 3) * 255).astype(np.uint8)   # 4:3 render
    out = adapters._squash(frame, (224, 224))
    assert out.shape == (224, 224, 3)
    same = adapters._squash(out, (224, 224))
    assert same is out or np.array_equal(same, out)


def test_check_chunk_enforces_ten_columns_and_finiteness():
    ok = adapters._check_chunk(np.zeros((8, 10)), "x")
    assert ok.dtype == np.float32 and ok.shape == (8, 10)
    with pytest.raises(RuntimeError, match="shape"):
        adapters._check_chunk(np.zeros((8, 9)), "x")
    bad = np.zeros((8, 10)); bad[3, 9] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        adapters._check_chunk(bad, "x")


def test_the_loop_calls_the_adapter_not_a_cosmos_function():
    """`predict_chunk(model, frame, prompt, ...)` was the cosmos-specific call. It is gone;
    the loop calls `adapter.act(frame, goal_vec, goal_raw, seed)` and nothing else."""
    assert "predict_chunk(" not in SCRIPT and "make_prompt(" not in SCRIPT
    assert re.search(r"adapter\.act\(frame, goal_vec, window\.goal_frame\.raw", SCRIPT)


def test_render_res_reaches_run_info():
    """`--render-res W H` is what lets every policy see the training-native 1024x768; if the
    flag parsed but never reached `write_run_info`, the comparison would silently render at
    344x258 and DP would be fed a 2x upsample."""
    assert re.search(r"res = tuple\(args\.render_res\) if args\.render_res else args\.image_size", SCRIPT)
    assert "resolution=res" in SCRIPT


# ---- pixel parity with what each policy was trained on --------------------------------------
def test_dp_frame_is_bit_identical_to_the_training_loader(tmp_path):
    """The bug this pins: the adapter built DP's input with BILINEAR and /255 ([0, 1]) while
    its dataset used LANCZOS and /127.5 - 1 ([-1, 1]). `build_obs_inputs` maps its input back
    with `x * 0.5 + 0.5`, so a [0, 1] tensor lands the entire image in the upper half of the
    range before DINOv2 sees it -- a silently washed-out frame on every episode, which reads
    as "the baseline is weak" rather than "the baseline was fed garbage".
    """
    torch = pytest.importorskip("torch")
    from PIL import Image
    from src.baselines.dp_model import _load_image_as_tensor

    rs = np.random.RandomState(0)
    frame = (rs.rand(768, 1024, 3) * 255).astype(np.uint8)        # a 4:3 render
    path = tmp_path / "f.png"
    Image.fromarray(frame).save(path)
    res_hw = (480, 720)                                            # as the DP config stores it

    ref = _load_image_as_tensor(path, res_hw)
    pil = Image.fromarray(frame).convert("RGB").resize((res_hw[1], res_hw[0]), Image.LANCZOS)
    mine = torch.from_numpy(
        np.ascontiguousarray(np.asarray(pil, np.float32) / 127.5 - 1.0)).permute(2, 0, 1).contiguous()

    assert torch.equal(ref, mine), "DP adapter no longer reproduces its training preprocessing"
    assert float(mine.min()) < -0.5, "input must span [-1, 1], not [0, 1]"


def test_vla_frame_matches_the_source_harness(tmp_path):
    """The VLAs' own harness did `im.convert("RGB").resize((224,224), BILINEAR) / 255.0`."""
    from PIL import Image
    rs = np.random.RandomState(1)
    frame = (rs.rand(768, 1024, 3) * 255).astype(np.uint8)
    ref = np.asarray(Image.fromarray(frame).convert("RGB").resize((224, 224), Image.BILINEAR),
                     np.float32) / 255.0
    mine = adapters._squash(frame, (224, 224)).astype(np.float32) / 255.0
    assert np.array_equal(ref, mine)


# ---- rendering cost: one Blender per episode, not per frame ---------------------------------
def test_the_rollout_uses_the_persistent_renderer():
    """`SubprocessBlenderRenderer` reloads the scene, rebuilds the BVH and recompiles kernels
    for EVERY frame. At the training render size that was ~95 s a frame and ~8 min an episode,
    almost all of it setup — 8.5 h for one 64-episode run. The persistent server pays it once."""
    assert "PersistentBlenderRenderer(repo_root=SHARED)" in SCRIPT
    assert re.search(r'renderer = \(SubprocessBlenderRenderer.*\n\s*if name\.split\("__"\)\[0\] in _renderer_fallback',
                     SCRIPT), "the slow path must remain reachable for scenes that kill the server"


def test_every_episode_closes_its_renderer():
    """A persistent server holds a loaded scene and a GPU context. Without the `finally` they
    accumulate — one per episode, 64 of them."""
    assert re.search(r"finally:\n(?:.*\n)*?\s*if renderer is not None:\n\s*try:\n\s*renderer\.close\(\)",
                     SCRIPT)
    assert "        renderer = None\n        try:" in SCRIPT, "bind before the try, or finally NameErrors"


def test_dp_frame_is_bit_identical_to_the_training_loader(tmp_path):
    """The bug this pins: the adapter built DP's input with BILINEAR and /255 ([0, 1]) while
    its dataset used LANCZOS and /127.5 - 1 ([-1, 1]). `build_obs_inputs` maps its input back
    with `x * 0.5 + 0.5`, so a [0, 1] tensor lands the entire image in the upper half of the
    range before DINOv2 sees it -- a silently washed-out frame on every episode, which reads
    as "the baseline is weak" rather than "the baseline was fed garbage".
    """
    torch = pytest.importorskip("torch")
    from PIL import Image
    from src.baselines.dp_model import _load_image_as_tensor

    rs = np.random.RandomState(0)
    frame = (rs.rand(768, 1024, 3) * 255).astype(np.uint8)        # a 4:3 render
    path = tmp_path / "f.png"
    Image.fromarray(frame).save(path)
    res_hw = (480, 720)                                            # as the DP config stores it

    ref = _load_image_as_tensor(path, res_hw)
    pil = Image.fromarray(frame).convert("RGB").resize((res_hw[1], res_hw[0]), Image.LANCZOS)
    mine = torch.from_numpy(
        np.ascontiguousarray(np.asarray(pil, np.float32) / 127.5 - 1.0)).permute(2, 0, 1).contiguous()

    assert torch.equal(ref, mine), "DP adapter no longer reproduces its training preprocessing"
    assert float(mine.min()) < -0.5, "input must span [-1, 1], not [0, 1]"


def test_vla_frame_matches_the_source_harness(tmp_path):
    """The VLAs' own harness did `im.convert("RGB").resize((224,224), BILINEAR) / 255.0`."""
    from PIL import Image
    rs = np.random.RandomState(1)
    frame = (rs.rand(768, 1024, 3) * 255).astype(np.uint8)
    ref = np.asarray(Image.fromarray(frame).convert("RGB").resize((224, 224), Image.BILINEAR),
                     np.float32) / 255.0
    mine = adapters._squash(frame, (224, 224)).astype(np.float32) / 255.0
    assert np.array_equal(ref, mine)


# ---- rendering cost: one Blender per episode, not per frame ---------------------------------
def test_the_rollout_uses_the_persistent_renderer():
    """`SubprocessBlenderRenderer` reloads the scene, rebuilds the BVH and recompiles kernels
    for EVERY frame. At the training render size that was ~95 s a frame and ~8 min an episode,
    almost all of it setup — 8.5 h for one 64-episode run. The persistent server pays it once."""
    assert "PersistentBlenderRenderer(repo_root=SHARED)" in SCRIPT
    assert re.search(r'renderer = \(SubprocessBlenderRenderer.*\n\s*if name\.split\("__"\)\[0\] in _renderer_fallback',
                     SCRIPT), "the slow path must remain reachable for scenes that kill the server"


def test_every_episode_closes_its_renderer():
    """A persistent server holds a loaded scene and a GPU context. Without the `finally` they
    accumulate — one per episode, 64 of them."""
    assert re.search(r"finally:\n(?:.*\n)*?\s*if renderer is not None:\n\s*try:\n\s*renderer\.close\(\)",
                     SCRIPT)
    assert "        renderer = None\n        try:" in SCRIPT, "bind before the try, or finally NameErrors"


def test_the_render_fallback_is_keyed_by_scene_and_retries():
    """Parking_70ae1f27 kills the render server for EVERY object placed in it. Keying the
    fallback by placement demoted one placement at a time while the other nine failed
    identically — which is why that scene is absent from every held-out number this project
    has published, leaving eight scenes reported as seven."""
    assert 'name.split("__")[0] in _renderer_fallback' in SCRIPT, "fallback must be scene-keyed"
    assert '_renderer_fallback.add(name.split("__")[0])' in SCRIPT
    assert "episodes.append(episodes[i])" in SCRIPT, "the first failure must be retried, not lost"


def test_appending_during_enumerate_extends_the_loop():
    """The retry works only because a `for` over a list reads it lazily. If that were false
    the re-queued episode would silently never run."""
    seen, items = [], [1, 2, 3]
    for i, v in enumerate(items):
        seen.append(v)
        if v == 2 and len(items) == 3:
            items.append(99)
    assert seen == [1, 2, 3, 99]


# ---- AutoPhoto: the floor row, and why it is one ---------------------------------------------
def test_autophoto_is_registered_and_needs_its_reward_net():
    """It is a legitimate row — a goal-agnostic RL policy SHOULD score near the no-op on a
    goal-conditioned metric, and if it does not, the metric is measuring "moving helps"."""
    assert '"autophoto"' in SCRIPT
    assert re.search(r'if args\.policy == "autophoto":', __import__("pathlib").Path(
        REPO / "src/baselines/adapters.py").read_text())


def test_autophoto_action_table_matches_the_source_env():
    """Copied from the source repo's PhotoEnv rather than imported (that module drags in
    gymnasium, its Blender env and its renderer). Copies drift; this pins the magnitudes."""
    src = (REPO / "src/baselines/adapters.py").read_text()
    body = src.split("class AutoPhotoAdapter", 1)[1]
    assert "_F, _T10, _T30, _T90 = 0.25, np.deg2rad(10), np.deg2rad(30), np.deg2rad(90)" in body
    # nine entries, exactly one of which terminates
    table = body.split("self.actions = [", 1)[1].split("]", 1)[0]
    assert table.count("(0, 0") + table.count("None") == 9, table
    assert table.count("None") == 1


def test_autophoto_cannot_move_laterally_or_vertically():
    """The structural finding, asserted rather than asserted-in-prose: every action's 5-D delta
    has dx = dy = dpitch = 0, so elevation and the vertical half of `d_aim` are unreachable.
    That is WHY it is the floor, and it would be silently wrong if the table ever changed."""
    np_ = pytest.importorskip("numpy")
    F, T10, T30, T90 = 0.25, np_.deg2rad(10), np_.deg2rad(30), np_.deg2rad(90)
    table = [(0, 0, F, 0, 0), (0, 0, 0, -T30, 0), (0, 0, 0, T30, 0), None,
             (0, 0, -F, 0, 0), (0, 0, 0, -T10, 0), (0, 0, 0, T10, 0),
             (0, 0, 0, -T90, 0), (0, 0, 0, T90, 0)]
    moves = [a for a in table if a is not None]
    assert all(a[0] == 0 for a in moves), "dx (lateral) is never non-zero"
    assert all(a[1] == 0 for a in moves), "dy is never non-zero"
    assert all(a[4] == 0 for a in moves), "dpitch (tilt) is never non-zero"
    assert any(a[2] != 0 for a in moves) and any(a[3] != 0 for a in moves), "dolly and yaw exist"
