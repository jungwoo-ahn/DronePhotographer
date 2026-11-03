"""Laptop-side drone control for PX4 SITL.

Deliberately self-contained: this package is the only thing that has to run on the
laptop next to PX4, so it must import without Blender, torch, Cosmos or anything else
in `src/`. The policy lives on the GPU server and is reached over HTTP; see
`policy_client`. Keep it that way -- if you find yourself importing `src.policy` here,
move the shared piece into `commands.py` instead.
"""
