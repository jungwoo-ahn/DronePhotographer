"""Inspect a Blender scene for 'fake asset' signals.

Usage:
  <blender_binary> --background <scene.blend> --python data/blender_inspect.py

Or:
  bash -c '/path/to/blender --background path/scene.blend --python data/blender_inspect.py'

Prints a JSON verdict to stdout on a line prefixed ###INSPECT### so it can be
parsed out of Blender's noisy output.

Verdict is "fake" only when BOTH of these fire together:
  - total_polys < 1000    (not enough geometry to be a real 3D scene)
  - mesh_count <= 3       (few distinct objects — consistent with a
                           billboard plane + maybe a light/camera)

Either signal alone produces false positives (stylized low-poly models;
scenes dominated by one terrain plane). Billboard/image-plane assets
reliably trip both.
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

    # Fake = billboard/image-plane. Require BOTH very low poly count AND few
    # meshes — either alone produces too many false positives (a stylized
    # low-poly model; a legit scene with a big terrain plane; a flat floor plan).
    low_poly = total_polys < 1000
    few_meshes = len(meshes) <= 3
    signals = []
    if low_poly:
        signals.append(f"low_poly_count({total_polys})")
    if few_meshes:
        signals.append(f"few_meshes({len(meshes)})")

    verdict = "fake" if (low_poly and few_meshes) else "ok"
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
