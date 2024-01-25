###########################################
# Render videos from training trajectory  #
###########################################

# tnt
# python render.py --config configs/tnt_playground.txt --render_train

# # 360
# python render.py --config configs/360_bonsai.txt --render_train
# python render.py --config configs/360_counter.txt --render_train
# python render.py --config configs/360_garden.txt --render_train

# # LERF
# python render.py --config configs/lerf_donuts.txt --render_train
# python render.py --config configs/lerf_dozer_nerfgun_waldo.txt --render_train
# python render.py --config configs/lerf_espresso.txt --render_train
# python render.py --config configs/lerf_figurines.txt --render_train
# python render.py --config configs/lerf_ramen.txt --render_train
# python render.py --config configs/lerf_shoe_rack.txt --render_train
# python render.py --config configs/lerf_teatime.txt --render_train
# python render.py --config configs/lerf_waldo_kitchen.txt --render_train


###########################################
# Render videos from custom trajectory    #
###########################################

# LERF
python render.py --config configs/lerf_teatime.txt --render_traj="/home/max/Documents/maxhsu/datasets/lerf/teatime/custom_camera_path/transforms_001.json"
python render.py --config configs/tnt_playground.txt --render_traj="/home/max/Documents/maxhsu/datasets/tnt/Playground/custom_camera_path/transforms_001.json"
python render.py --config configs/360_counter.txt --render_traj="/home/max/Documents/maxhsu/datasets/360/counter/custom_camera_path/transforms_001.json"
python render.py --config configs/360_garden.txt --render_traj="/home/max/Documents/maxhsu/datasets/360/garden/custom_camera_path/transforms_001.json"