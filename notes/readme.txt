1. render images
    CUDA_VISIBLE_DEVICES=6 blender -b -P render_object.py -- \
        --input_scene assets/Koky_LuxuryHouse_1.blend \
        --camera_direction_offsets 10 10 10 

2. score images ( metric = brightness, laplacian, stddev | qalign, siglip2 )

    python score_images.py --output_dir outputs/Koky_LuxuryHouse_1_251222_105417 --metric siglip2

3. visualize stats, scores, images
stats:
    python tools/visualize_stats.py --output_dir outputs/Koky_LuxuryHouse_1_251222_105417
    (optional: --metric qalign)
rankings:
    python tools/visualize_rankings.py --output_dir outputs/Koky_LuxuryHouse_1_251222_105417
    (optional: --metric qalign --limit 20 )

4. filter images
    python filter_images.py --output_dir outputs/Koky_LuxuryHouse_0_251222_105729
    (optional: --limit 30)


Koky_LuxuryHouse_0_251222_105729
Koky_LuxuryHouse_1_251222_105417
Koky_LuxuryHouse_2_251222_105439