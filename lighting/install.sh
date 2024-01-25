# This script provides a way to setup the environment for Text2Light.

conda create -n text2light python=3.8
conda activate text2light

# install Pytorch and cudatoolkit
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# install other dependencies
pip install opencv-python
pip install tqdm
pip install tensorboardX
pip install termcolor

export OPENCV_IO_ENABLE_OPENEXR=true