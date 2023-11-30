# 360 (garden)
CUDA_LAUNCH_BLOCKING=1 python demo/demo_with_text.py \
  --chunk_size 4 \
  --img_path ../datasets/360/garden/images/ \
  --amp --temporal_setting semionline \
  --output /home/max/Desktop/instant-ngp-pp/results/colmap/garden/track_with_deva/ \
  --prompt "table" \
  --DINO_THRESHOLD 0.6