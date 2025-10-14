#!/bin/bash

conda create -n habitat3.0 python=3.9 cmake=3.14.0
conda config --set channel_priority flexible
conda install -y \
      -c conda-forge -c defaults -c nvidia/label/cuda-12.4.1 \
      cuda=12.4.1 cudnn gcc_linux-64=13.2.0 gxx_linux-64=13.2.0

conda install ninja -y


################## set system var #########################
export CMAKE_GENERATOR=Ninja
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"

export LDFLAGS="-L/usr/lib/x86_64-linux-gnu ${LDFLAGS}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH}"
sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CONDA_PREFIX/lib/libcuda.so"
export TORCH_CUDA_ARCH_LIST="8.6"


########## install habitat3.0 ########################

conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat
pip install opencv-python==4.11.0.86
pip install pyyaml
pip install yapf
pip install mmengine==0.7.3
conda clean -afy
pip cache purge

########## install pointcept ########################

# add NVIDIA channel because pytorch-cuda lives there
conda install -y \
      pytorch=2.5.0 torchvision=0.20.0 torchaudio=2.5.0 pytorch-cuda=12.4 \
      -c pytorch -c nvidia -c conda-forge
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu124

# PPT (clip)
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# flash-attn

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git fetch --tags
git checkout v2.6.0
pip install packaging
python -m pip install -v --no-cache-dir --no-build-isolation --no-binary :all: .

#export TORCH_CUDA_ARCH_LIST="80;86;89;90"
export TORCH_CUDA_ARCH_LIST="8.0"
export FLASH_ATTENTION_FORCE_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"

# PTv1 & PTv2 or precise eval
cd ..
cd pointops
python setup.py install

cd ..
cd pointops2
python setup.py install

########## install ActiveSGM ########################

# tiny-cuda-nn
pip install --no-build-isolation --no-cache-dir \
  'git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch'
pip install charset_normalizer==2.0.4

# pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# gaussain render with depth
git clone https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git
cd diff-gaussian-rasterization-w-depth
git checkout cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110
sed -i '1i #include <cstddef>\n#include <cstdint>' cuda_rasterizer/rasterizer_impl.h
pip install --no-build-isolation --no-cache-dir .

# channel render
cd ..
git clone https://github.com/lly00412/semantic-gaussians.git
cd semantic-gaussians/
git checkout liyan/dev

# modify config.h base on number of class
#NUM_CHANNELS {num of class} // Default 3

d channel-rasterization/
sed -i '1i #include <cstdint>\n#include <cstddef>' cuda_rasterizer/rasterizer_impl.h
pip install --no-build-isolation --no-cache-dir -v .

# sparse render
cd ../..
git clone https://github.com/lly00412/semantic-gaussians.git ./sparse_render
cd sparse_render
git checkout hairong/sparse_ver
cd sparse-channel-rasterization/
sed -i '1i #include <cstdint>\n#include <cstddef>' cuda_rasterizer/rasterizer_impl.h

# modify config.h base on number of class
#NUM_CHANNELS {num of class} // Default 3 MP3D:41


############## install other reqirement ##########################

cd ../..
pip install -r requirements.txt
pip install -r requirements_semantic.txt
