"""GPU configuration for Blender Cycles rendering."""

from __future__ import annotations


def configure_gpu(scene, disable_gpu=False, gpu_backend="AUTO", gpu_devices=None):
    """Configure GPU rendering for Cycles. Must run inside Blender.

    Returns dict with 'device' and 'devices' keys. Raises RuntimeError if GPU
    init fails (we never silently fall back to CPU rendering).
    """
    import bpy

    scene.render.engine = "CYCLES"
    if disable_gpu:
        scene.cycles.device = "CPU"
        print("GPU_CONFIG: CPU (disable_gpu=True)")
        return {"device": "CPU", "devices": []}
    cprefs_addon = bpy.context.preferences.addons.get("cycles")
    if cprefs_addon is None:
        raise RuntimeError("cycles addon not loaded; cannot configure GPU")
    cprefs = cprefs_addon.preferences
    backend_order = ("OPTIX", "CUDA") if gpu_backend == "AUTO" else (gpu_backend,)
    last_err = None
    for dtype in backend_order:
        try:
            cprefs.compute_device_type = dtype
            cprefs.get_devices()
        except Exception as e:
            last_err = e
            print(f"GPU_CONFIG: backend {dtype} init failed: {e}")
            continue
        backend_devices = [d for d in cprefs.devices if d.type == dtype]
        if not backend_devices:
            print(f"GPU_CONFIG: backend {dtype} found 0 devices")
            continue
        selected_indices = set()
        if gpu_devices:
            valid = [idx for idx in gpu_devices if 0 <= idx < len(backend_devices)]
            selected_indices = set(valid)
        for d in cprefs.devices:
            d.use = False
        for idx, d in enumerate(backend_devices):
            d.use = idx in selected_indices if selected_indices else True
        active = [d.name for d in backend_devices if d.use]
        if not active:
            print(f"GPU_CONFIG: backend {dtype} matched 0 active devices "
                  f"(gpu_devices={gpu_devices}, visible={len(backend_devices)})")
            continue
        scene.cycles.device = "GPU"
        print(f"GPU_CONFIG: device={dtype} active={active}")
        return {"device": dtype, "devices": active}
    raise RuntimeError(
        f"No GPU backend available (tried {backend_order}); "
        f"refusing to fall back to CPU. Last error: {last_err}"
    )


def gpu_config_snippet(gpu_backend: str = "AUTO", gpu_devices: list[int] | None = None) -> str:
    """Return inline Python code string for GPU config inside Blender subprocesses.

    Use this when launching Blender via subprocess with --python-expr.
    """
    devices_literal = repr(gpu_devices) if gpu_devices else "None"
    return f"""
# --- GPU configuration ---
_gpu_backend = {gpu_backend!r}
_gpu_devices = {devices_literal}

if _gpu_backend == "CPU":
    bpy.context.scene.cycles.device = "CPU"
else:
    _cprefs_addon = bpy.context.preferences.addons.get("cycles")
    if _cprefs_addon is None:
        bpy.context.scene.cycles.device = "CPU"
    else:
        _cprefs = _cprefs_addon.preferences
        _backend_order = ("OPTIX", "CUDA") if _gpu_backend == "AUTO" else (_gpu_backend,)
        _gpu_ok = False
        for _dtype in _backend_order:
            try:
                _cprefs.compute_device_type = _dtype
                _cprefs.get_devices()
            except Exception:
                continue
            _backend_devices = [d for d in _cprefs.devices if d.type == _dtype]
            if not _backend_devices:
                continue
            _selected = set()
            if _gpu_devices:
                _selected = {{idx for idx in _gpu_devices if 0 <= idx < len(_backend_devices)}}
            for d in _cprefs.devices:
                d.use = False
            for _idx, d in enumerate(_backend_devices):
                d.use = _idx in _selected if _selected else True
            bpy.context.scene.cycles.device = "GPU"
            _gpu_ok = True
            print(f"GPU render: {{_dtype}} — {{[d.name for d in _backend_devices if d.use]}}")
            break
        if not _gpu_ok:
            bpy.context.scene.cycles.device = "CPU"
# --- end GPU configuration ---
"""
