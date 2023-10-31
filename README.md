# Improved instant-NGP model
This repo is an improved version of another [Instant-NGP repo](https://github.com/kwea123/ngp_pl), and bases on pytorch implementation. 

# Dependencies

So far this code has been tested on lab machine (GPU 4070 Ti and CUDA 11.7).

## Python libraries

```
conda create -n ngp python=3.8
conda activate ngp
python -m pip install --upgrade pip
pip uninstall torch torchvision functorch tinycudann

### PyTorch ###
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117

### Install CUDA 11.7 toolkit (if needed) ###
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit

### Pytorch-scatter ###
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

### Required Packages ###
pip install -r requirements.txt

### Objaverse API ###
pip install --upgrade --quiet objaverse
```

## TinyCudaNN

This repo relies on TinyCudaNN.

Installation steps are as follows:

1. git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git
2. cd tiny-cuda-nn
3. Use your favorite editor to edit `include/tiny-cuda-nn/common.h` and set `TCNN_HALF_PRECISION` to `0` (see https://github.com/NVlabs/tiny-cuda-nn/issues/51 for details)
4. cd bindings/torch
5. python setup.py install

## Compile CUDA extension of this project

Run `pip install models/csrc/` (please run this each time you `pull` the code)

## OpenAI API Module

* `pip install openai`
* Register for an API key and run `export OPENAI_API_KEY=<set-your-key-here>`

# Preparing data

- **If you want to test scene editing directly without training NeRF and extracting semantic meshes, we provide the required data and procedure at the bottom. (currently only Playground scene in Tanks & Temples are available)**
- Download preprocessed data at https://drive.google.com/drive/folders/1hLIZNTTB_6jeo38JgjDY-9jb_2M1I05E?usp=sharing (LERF, MipNeRF360, Tanks & Temples)
- Create a folder and put it under `../datasets/<dataset-name>/<scene-name>`, can refer to root_dir in config files

# Run training!

```
python train.py --config configs/tnt_playground.txt
```

This code will validate your model when training procedure finishes.

# Resume training!

```
python train.py --config configs/Playground.txt --ckpt_path PATH/TO/CHECKPOINT/DIR/epoch={n}.ckpt
```

There is a bug of pytorch lightning regarding to progress bar(see https://github.com/Lightning-AI/lightning/issues/13124 for details). 

# Renderings

```
python render.py --config configs/tnt_playground.txt --weight_path PATH/TO/SLIM/CHECKPOINT/DIR/epoch={n}_slim.ckpt
```

# Extract semantic meshes

This part we use Tracking-with-DEVA + voting schemes. **##TODO##**

# Run scene editing

## Prepare required data & ckpts

* Download `playground_no_semantic_ckpts.zip`, `playground_no_semantic.zip` and `tnt.zip` from https://drive.google.com/drive/folders/1Ul6au8YSUlJEcLR88zXR0TnH6pwgRrIO?usp=sharing.
* Extract `tnt.zip` and place it under `../datasets/`.
* Extract `playground_no_semantic_ckpts.zip` and place it under `./ckpts/tnt/`.
* Extract `playground_no_semantic.zip` and place it under `./results/tnt/`

## Run editing script

* Run `python edit_scene.py --config=configs/tnt_playground.txt --edit_text="Place an apple on the sand."`.
* Note that the response is fixed for debugging purpose. You can simply delete the fixed response and uncomment the following two lines.
    ```
    # Generate the code
    # edit_text = hparams.edit_text
    # response = generate_response(edit_text)
    ```

* To skip re-rendering NeRF views, you can set `skip_render_NeRF=True` in `edit_utils.py`.
* There might be error that codes generated from GPT-4 model is not executable since the format of output has error. Need further careful prompting improvement. **##TODO##**