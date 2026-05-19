"""Sky/world environment setup for Blender."""

from __future__ import annotations

import math
import random

import bpy


def set_nishita_sky(strength: float = 0.3, randomize_sun: bool = True) -> None:
    """Set Nishita sky as world background.

    Args:
        strength: Background shader strength.
        randomize_sun: If True, randomize sun elevation and rotation.
    """
    if strength <= 0:
        return

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    bg = tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = float(strength)
    sky = tree.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"

    if randomize_sun:
        sky.sun_elevation = math.radians(random.uniform(15, 75))
        sky.sun_rotation = math.radians(random.uniform(0, 360))

    output = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(sky.outputs["Color"], bg.inputs["Color"])
    tree.links.new(bg.outputs["Background"], output.inputs["Surface"])
