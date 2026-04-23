"""Inspect a Blender scene for 'fake asset' signals.

Usage:
  <blender_binary> --background <scene.blend> --python data/blender_inspect.py

Or:
  bash -c '/path/to/blender --background path/scene.blend --python data/blender_inspect.py'

Prints a JSON verdict to stdout on a line prefixed ###INSPECT### so it can be
parsed out of Blender's noisy output.

Heuristics for "fake":
  - total_polys < 1000                        (not enough geometry)
  - mesh_count <= 3                           (few distinct objects)
  - z_extent / max(x_extent, y_extent) < 0.05 (basically flat)
  - one mesh accounts for > 80% of total polys AND has a large XY footprint
    relative to the scene (likely a billboard/background plane)
"""
import bpy
import json
import sys
from mathutils import Vector


def axis_extent(verts_world):
    if not verts_world:
        return Vector((0, 0, 0))
    mn = Vector(verts_world[0])
    mx = Vector(verts_world[0])
    for v in verts_world[1:]:
        mn.x = min(mn.x, v.x); mn.y = min(mn.y, v.y); mn.z = min(mn.z, v.z)
        mx.x = max(mx.x, v.x); mx.y = max(mx.y, v.y); mx.z = max(mx.z, v.z)
    return mx - mn


def main():
    scene = bpy.context.scene
    meshes = [o for o in scene.objects if o.type == "MESH"]
    if not meshes:
        print(f"###INSPECT### {json.dumps({'verdict': 'fake', 'reason': 'no meshes'})}", flush=True)
        return

    total_polys = 0
    per_mesh = []  # (name, polys, xy_area)
    all_corners = []
    for o in meshes:
        mesh = o.data
        if not mesh:
            continue
        npolys = len(mesh.polygons)
        total_polys += npolys
        # World-space bbox corners (8 corners of local bbox transformed by world matrix)
        corners = [o.matrix_world @ Vector(c) for c in o.bound_box]
        all_corners.extend(corners)
        ext = axis_extent(corners)
        xy_area = abs(ext.x) * abs(ext.y)
        per_mesh.append((o.name, npolys, xy_area, ext))

    scene_ext = axis_extent(all_corners)
    sx, sy, sz = abs(scene_ext.x), abs(scene_ext.y), abs(scene_ext.z)
    max_xy = max(sx, sy)
    flatness = sz / max_xy if max_xy > 0 else 0.0
    scene_xy_area = sx * sy

    # Find single dominant polygon-hog (billboard candidate)
    per_mesh.sort(key=lambda t: -t[1])
    biggest = per_mesh[0] if per_mesh else None
    poly_share = (biggest[1] / total_polys) if (biggest and total_polys) else 0.0
    footprint_share = (biggest[2] / scene_xy_area) if (biggest and scene_xy_area > 0) else 0.0

    signals = []
    if total_polys < 1000:
        signals.append(f"low_poly_count({total_polys})")
    if len(meshes) <= 3:
        signals.append(f"few_meshes({len(meshes)})")
    if flatness < 0.05:
        signals.append(f"flat(z/xy={flatness:.3f})")
    if biggest and poly_share > 0.8 and footprint_share > 0.5:
        signals.append(
            f"dominant_plane({biggest[0]}: {poly_share:.0%} polys, {footprint_share:.0%} footprint)"
        )

    verdict = "fake" if signals else "ok"
    report = {
        "verdict": verdict,
        "signals": signals,
        "mesh_count": len(meshes),
        "total_polys": total_polys,
        "scene_extent_xyz": [round(sx, 2), round(sy, 2), round(sz, 2)],
        "flatness_z_over_xy": round(flatness, 4),
        "top_meshes_by_polys": [
            {"name": n, "polys": p, "xy_area": round(a, 2)}
            for n, p, a, _ in per_mesh[:5]
        ],
    }
    print(f"###INSPECT### {json.dumps(report)}", flush=True)


if __name__ == "__main__":
    main()
