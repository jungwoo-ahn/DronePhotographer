#!/bin/bash
# Render best 25 scenes, 100 images each, multi-GPU
# Usage: bash render_best25.sh
# Change GPU_DEVICES to use different/fewer GPUs

GPU_DEVICES="1 2 3 4 5 6 7"

python scripts/render_objects_in_multiple_scenes.py \
    --placements placements_best25.json \
    --assets_root /home/nas5/jungwooahn/datasets/DronePhotos/assets \
    --output_dir outputs \
    --num_images_per_placement 100 \
    --gpu_devices $GPU_DEVICES \
    --camera_radius_range 1.5 9 \
    --camera_direction_offsets 20 20 0 \
    --hemisphere \
    --adaptive_sampling \
    --samples 32 \
    --use_aabb_center
