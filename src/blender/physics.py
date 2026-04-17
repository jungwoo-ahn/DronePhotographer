"""Rigid body physics for settling objects onto surfaces."""

from __future__ import annotations

import bpy
from mathutils import Vector


def settle_object(root, imported, scene_meshes, sim_frames=60, substeps=20):
    """Drop an object and let it settle on scene surfaces using rigid body physics.

    Args:
        root: The root empty parenting the imported objects.
        imported: List of imported bpy mesh objects.
        scene_meshes: List of scene mesh objects (surfaces).
        sim_frames: Number of frames to simulate (at 24fps, 60 = 2.5 sec).
        substeps: Physics substeps per frame for accuracy.

    Returns:
        Final position as [x, y, z], or None on failure.
    """
    scene = bpy.context.scene

    # Store original frame range
    orig_start = scene.frame_start
    orig_end = scene.frame_end
    orig_current = scene.frame_current

    try:
        # Setup simulation range
        scene.frame_start = 1
        scene.frame_end = sim_frames
        scene.frame_current = 1

        # Enable rigid body world
        if scene.rigidbody_world is None:
            bpy.ops.rigidbody.world_add()
        rb_world = scene.rigidbody_world
        rb_world.substeps_per_frame = substeps
        rb_world.solver_iterations = 10
        rb_world.point_cache.frame_start = 1
        rb_world.point_cache.frame_end = sim_frames

        # Make scene meshes passive rigid bodies (static surfaces)
        for obj in scene_meshes:
            if obj.rigid_body is not None:
                continue
            try:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_add(type='PASSIVE')
                obj.rigid_body.collision_shape = 'MESH'
                obj.rigid_body.friction = 0.8
                obj.select_set(False)
            except Exception:
                obj.select_set(False)
                continue

        # Make imported objects active rigid bodies (will fall with gravity)
        for obj in imported:
            if obj.type != 'MESH':
                continue
            try:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_add(type='ACTIVE')
                obj.rigid_body.collision_shape = 'CONVEX_HULL'
                obj.rigid_body.mass = 1.0
                obj.rigid_body.friction = 0.8
                obj.rigid_body.linear_damping = 0.4
                obj.rigid_body.angular_damping = 0.9  # prevent tumbling
                obj.select_set(False)
            except Exception:
                obj.select_set(False)
                continue

        # Lift object before simulation so it can fall and settle
        start_z = root.location.z
        root.location.z += 0.3
        bpy.context.view_layer.update()

        # Step through frames to simulate
        for frame in range(1, sim_frames + 1):
            scene.frame_set(frame)

        # Read final position from last frame
        bpy.context.view_layer.update()
        final_pos = [root.location.x, root.location.y, root.location.z]

        # Validate: check object didn't fall through or embed in floor
        from src.blender.bbox import get_world_bbox
        final_bbox = get_world_bbox(imported)
        if final_bbox is None:
            print("Physics: lost object, reverting")
            return None

        # Check if fell more than 1m below starting position
        if final_bbox[0].z < start_z - 1.0:
            print(f"Physics: object fell too far (z={final_bbox[0].z:.2f}), reverting")
            return None

        # Post-settle validation: check object didn't embed in the floor.
        # Only push up slightly if the object is just barely below surface
        # (within 10cm). Large discrepancies mean physics went wrong — revert.
        if abs(final_bbox[0].z - start_z) > 0.5:
            print(f"Physics: settled too far from target "
                  f"(bottom={final_bbox[0].z:.2f}, target={start_z:.2f}), reverting")
            return None

        print(f"Physics: settled at z={final_pos[2]:.3f}")
        return final_pos

    except Exception as e:
        print(f"Physics simulation failed: {e}")
        return None

    finally:
        # Clean up rigid body properties
        for obj in imported:
            if obj.rigid_body is not None:
                try:
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.rigidbody.object_remove()
                    obj.select_set(False)
                except Exception:
                    obj.select_set(False)
        for obj in scene_meshes:
            if obj.rigid_body is not None:
                try:
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.rigidbody.object_remove()
                    obj.select_set(False)
                except Exception:
                    obj.select_set(False)

        # Restore frame range
        scene.frame_start = orig_start
        scene.frame_end = orig_end
        scene.frame_current = orig_current
        bpy.context.view_layer.update()
