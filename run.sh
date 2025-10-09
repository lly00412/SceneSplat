#!/bin/bash


#
export PYTHONPATH=$PWD:$PYTHONPATH
#export CUDA_HOME="$CONDA_PREFIX"
#export PATH="$CUDA_HOME/bin:$PATH"
#export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64"
#
#export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
#export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
#export CUDAHOSTCXX="$CXX"
#export TORCH_CUDA_ARCH_LIST="80;86;89;90"
#export FLASH_ATTENTION_FORCE_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"

scenes=( GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG YmJkqBEsHnH)

#for selected_scene in ${scenes[@]}
#do

#python scripts/preprocess_gs_v2.py \
#       --input /mnt/Data2/liyan/ActiveSGM/results/MP3D/${selected_scene}/ActiveSem/run_0/splatam/final/params.npz \
#       --output /mnt/Data3/liyan/preprocess/MP3D/${selected_scene}/ActiveSGM/


#done

python tools/train.py \
    --config-file configs/concat_dataset/custom.py \
    --options skip_eval=True \
    save_feat=True\
    save_path=./eval/preprocess/MP3D \
    weight=/mnt/Data4/scene_splat_7k/checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth \
    test_only=True \
    --num-gpus 1