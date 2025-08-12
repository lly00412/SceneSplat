# SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining

<h2 align="left">ICCV 2025 Oral</h2>

[![arXiv](https://img.shields.io/badge/arXiv-2503.18052-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2503.18052)
[![GitHub](https://img.shields.io/badge/GitHub-Webpage-blue?logo=github)](https://github.com/unique1i/SceneSplat_webpage)
[![🤗 HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k)

<p align="left">
    <img src="media/vis.png" alt="SceneSplat Demo" style="width:100%; max-width:1200px;">
</p>

The implementation of our work: <strong>SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining</strong>. With the vision-language pretraining and the self-supervised training scheme, we unlock rich 3DGS semantic learning and introduce a generalizable, open-vocabulary 3DGS encoder that operates natively on 3DGS.
***

#### $^\star$[Yue Li](https://unique1i.github.io/)<sup>1</sup>, $^\star$[Qi Ma](https://qimaqi.github.io/)<sup>2,3</sup>, [Runyi Yang](https://runyiyang.github.io/)<sup>3</sup>, [Huapeng Li](https://scholar.google.com/citations?user=LkF7__QAAAAJ)<sup>2</sup>, [Mengjiao Ma](https://insait.ai/mengjiao-ma/)<sup>3,4</sup>, $^\dagger$[Bin Ren](https://amazingren.github.io/)<sup>3,5,6</sup>, [Nikola Popovic](https://insait.ai/dr-nikola-popovic/)<sup>3</sup>, [Nicu Sebe](https://disi.unitn.it/~sebe/)<sup>6</sup>, [Ender Konukoglu](https://people.ee.ethz.ch/~kender/)<sup>2</sup>, [Theo Gevers](https://staff.science.uva.nl/th.gevers/)<sup>1</sup>, [Luc Van Gool](https://insait.ai/prof-luc-van-gool/)<sup>2,3</sup>, [Martin R. Oswald](https://oswaldm.github.io/)<sup>1</sup>, and [Danda Pani Paudel](https://insait.ai/dr-danda-paudel/)<sup>3</sup>

$^\star$: Equal Contribution, $^\dagger$: Corresponding Author <br>

<sup>1</sup> University of Amsterdam <br>
<sup>2</sup> ETH Zürich <br>
<sup>3</sup> INSAIT <br>
<sup>4</sup> Nanjing University of Aeronautics and Astronautics <br>
<sup>5</sup> University of Pisa <br>
<sup>6</sup> University of Trento <br>

## Installation

Please set up the provided conda environment with Python 3.10, PyTorch 2.5.1, and CUDA 12.4. 

```bash
conda env create -f env.yaml
conda activate scene_splat
```

## Datasets
Our SceneSplat dataset is available in the data repository [SceneSplat-7K](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k) on Hugging Face. Before gaining access, please read and accept the terms for using the dataset [here](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k#dataset-license). 

Together with the 3DGS scenes, we also release the frames metadata that we used to optimize the 3DGS, and the 2D language features extracted during the 3DGS language label collection process. Please refer to the data repository for details.

## Data Preparation

The self-supervised pretraining part requires only the 3DGS scenes, while the vision-language pretraining requires additional language labels. During preparation, the [preprocessing](pointcept/datasets/preprocessing) folder provides the per-dataset preprocessing scripts to convert the original 3DGS scenes, after which the chunking script is used to chunk the scenes into smaller parts for training if necessary.

<details open>
<summary>For example, the following are the commands to preprocess the Matterport3D 3DGS scenes.</summary>

```bash
python -u preprocess_matterport3d_gs.py \
    --pc_root  /path/to/ptv3_preprocessed/matterport3d \
    --gs_root  /path/to/gaussian_world/matterport3d_region_mcmc_3dgs \
    --output_root /path/to/gaussian_world/preprocessed/matterport3d_region_mcmc_3dgs \
    --num_workers 8 \
    --feat_root /path/to/gaussian_world/matterport3d_region_mcmc_3dgs/language_features_siglip2

python -u sampling_chunking_data_gs.py \
    --dataset_root  /path/to/gaussian_world/preprocessed/matterport3d_region_mcmc_3dgs  \
    --output_dir /path/to/gaussian_world/preprocessed/matterport3d_region_mcmc_3dgs \
    --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers 12 --max_chunk_num 6 
```
</details>


We recommend using the already provided preprocessed 3DGS scenes for the pretraining pipeline. For each scene, the 3DGS `*.ply` files are stored in `*.npy` files by parameters, and 3DGS language labels are stored with `lang_feat.npy` and `valid_feat_mask.npy`.

```bash
.
├── color.npy
├── coord.npy
├── opacity.npy
├── quat.npy
├── scale.npy
├── lang_feat.npy
└── valid_feat_mask.npy
```
For the original 3DGS scenes, please download the preprocessed ScanNet, ScanNet++, and Matterport3D 3DGS scenes from the [data repository](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k), where you have to agree to the [License](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k#dataset-license) first. 

For preprocessed data, note only part of the uploaded folders are use for pretraining, please refer to [Preprocessed Pretraining Data](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k#preprocessed-language-pretraining-data) when downloading.

## Pretraining Configs

Every task configuration is stored in the `configs` folder. The configurations are organized by the dataset. Each configuration file contains all the necessary parameters for the pretraining process. 

- `_base_` - Base configurations for runtime and datasets
- `concat_dataset` - Multi-dataset joint training configurations
- `scannet`, `scannetpp`, `matterport3d` - Single dataset configurations

Please first adapt the dataset root paths within the config after downloading the dataset. 
```python
# Example
repo_root="/workspace/SceneSplat"
scannet_data_root = "/path/to/your/scannet_3dgs_mcmc_preprocessed"
scannetpp_data_root = "/path/to/your/scannetpp_v2_mcmc_3dgs"
matterport3d_data_root = "/path/to/your/matterport3d_region_mcmc_3dgs"
```

The configuration includes:
- **Model settings**: Model architecture, input channels, and layer configurations.
- **Training parameters**: Learning rates, batch sizes, number of workers.
- **Dataset configuration**: The train/validation/test data are in the `data['train']`, `data['val']`, and `data['test']` fields, respectively, together with the data augmentations we apply.
- **Evaluation settings**: Zero-shot semantic segmentation settings.


## Vision-Language Pretraining
**Train from scratch.** Note that the config settings can be passed on-the-fly using the `--options` flag. The `save_path` is the directory where the model checkpoints, logs and evaluation results will be saved. For example, the following commands start the run with the joint training config.
```bash
python tools/ssl_pretrain.py \
  --config-file configs/concat_dataset/ssl-pretrain-concat-scan-ppv2-matt-3rscan-arkit-hyper-mcmc-base.py \
  --options \
    save_path=exp_runs/ssl_pretrainer/ssl-pretrain-scannet-all-base \
    batch_size=$batch_size batch_size_val=$batch_size_val \
    batch_size_test=$batch_size_test num_worker=$num_worker gpu_nums=$gpu_num \
  --num-gpus $gpu_num \
```
**Multi-node training.**
The codebase supports multi-node training when submitted on a SLURM cluster with NCCL library. The bash script [here](sh_jobs/snellius/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive-nccl.sh) is an example to run the task on 4 nodes. The difference from the single-node training is that we use `srun` to launch the task and add the `--multi_node` flag. Note there might be cluster-specific settings for the NCCL config, e.g., `NCCL_SOCKET_IFNAME`. Please adapt the settings if necessary.

```bash
srun python tools/train.py \
    ......
    --multi_node
```

**Resume training from checkpoint.** You can resume training from a checkpoint by specifying the checkpoint path to the `weight` option and set `resume=True`. For example:
```bash
python tools/train.py \
    --config-file lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py \
    --options ......
    weight=model_best.pth \
    resume=True \
    --num-gpus $gpu_num \
```
If `weight` is set but not `resume`, the training will start from the checkpoint but not resume the training process, i.e., the epoch and scheduler will be reset.

**Evaluation during training.** Evaluation is run after every epoch on the specified `val` data. To speed up the process, evaluation is performed on Gaussians after grid sampling. Note that the evaluation always happens at the original dataset label locations and that neighboring voting is enabled. The best checkpoint is picked by the foreground mIoU metric. The following parameters are set for the evaluator:

```python
class_names='/path/to/class_names.txt', 
text_embeddings='/path/to/text_embeddings.pt',
excluded_classes=['wall', 'floor', 'ceiling'],  # Excluded classes
ignore_index=-1,
vote_k=25,
enable_voting=True,
confidence_threshold=0.1,
```

The pipeline supports adding multiple data for evaluation. For example, the config for joint training [here](configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py) adds three datasets for evaluation. When adding multiple data, please set the corresponding `class_names`, `text_embeddings` and `excluded_classes` for each dataset. 


**Testing.** The training config is set to run the testing process automatically after training with the `PreciseEvaluator` hook, which calls the tester. Different from the training evaluation, where grid sampling is applied, the testing process will split the 3DGS scene into chunks and ensure coverage of all input Gaussians. `ZeroShotSemSegTester` is used here, and it has similar parameters as the evaluator. 

```python
verbose=True,
class_names='/path/to/class_names.txt',
text_embeddings='/path/to/text_embeddings.pt',
excluded_classes=["wall", "floor", "ceiling"],
enable_voting=True,
vote_k=25,
confidence_threshold=0.1,
save_feat=False, # save the model inference features
skip_eval=False, # skip the metrics calculation
```

Testing also supports multiple datasets, please make sure the tester settings in `test` correspond to the `data['test']` in the config.
The evaluation results will be saved in the `save_path` directory.

To run the testing process individually, please add `test_only=True` to the `--options` flag, this will skip the training process all together and only run hooks that has `after_train()` attribute. The `weight` option should be set to the checkpoint path you want to test. 
```bash
python tools/train.py \
    --config-file lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py \
    --options ......
    weight=model_best.pth \
    test_only=True \
    --num-gpus $gpu_num \
```


**Reproducing the results.** We reported the best zero-shot results using the joint training [config](configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py), which covers 600 data epochs on the ScanNet, ScanNet++, and Matterport3D 3DGS scenes. The training takes around 3.5 days on 16 NVIDIA H100 GPUs using multi-node training. Note that the vision-language pretraining requires at least 48GB GPU memory, and the training results may vary for different runs. The checkpoint corresponding to this config is available [here](https://huggingface.co/GaussianWorld/SceneSplat_lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive), which should obtain the evaluation reported results for joint training.


## Self-Supervised Pretraining
Language features offer exciting opportunities for pretraining, though they can require long computation time on preprocessing. Building on the approach from [SimDINO](https://github.com/RobinWu218/SimDINO), we've developed an self-supervised pretraining method using parameters of gaussian splats only. The model checkpoint from self-supervised pretraining will be updated later as we are currently retraining our model on the expanded datasets.

Our approach differs from [Sonata](https://github.com/facebookresearch/sonata/tree/main/sonata) in several key aspects:

1. A **Gaussian-based multiview generator**
2. Additional **Masked Autoencoder (MAE) losses**
3. Use of the **SimDINO loss**, which is more stable and memory-efficient


### Data Preparation
To train the model in a self-supervised manner, the dataset should follow the structure described above—reusing the preprocessed data is acceptable. If compatibility is needed with PTv3 and Sonata, an additional normal attribute can be included during self-supervised training.

```bash
.
├── color.npy
├── coord.npy
├── opacity.npy
├── quat.npy
├── scale.npy
├── normal.npy (optional)
```

Please download more gaussians splats from our repo like hypersim, arkitssene, 3rscan etc for pretraining  [data repository](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k). After downloading, update the data path for each one in the [config](configs/concat_dataset/ssl-pretrain-concat-scan-ppv2-matt-3rscan-arkit-hyper-mcmc-base.py).


### Pretraining Details
We introduced an additional interface function and trainer to support **SimDINO-based self-supervised pretraining**. You can find the implementation in the [interface](tools/ssl_pretrain.py), [trainer](pointcept/engines/pretrain.py), and [losses](pointcept/models/losses) files.

The model follows the standard **PTv3 architecture**, consisting of:

* An **Encoder** that increases feature dimensions while reducing spatial resolution.
* A **Decoder** that restores spatial resolution and reduces feature dimensions, using skip connections.

We are also exploring ways to place more emphasis on the **Encoder**, similar to the approach used in [Sonata](https://github.com/facebookresearch/sonata/tree/main/sonata) with upcast attention.

The config differs from language pretraining in the following aspects:
* **Trainer**: We use [`DefaultSSLPreTrainer`](pointcept/engines/pretrain.py) instead of [`LangPretrainer`](pointcept/models/default.py).
* **Model**: We use [`PT-v3m1-simdino`](pointcept/models/point_transformer_v3_ssl/point_transformer_v3m1_ssl.py) instead of the standard [`PT-v3m1`](pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py).
* **Data**: We use the [`GenericGSDataset`](pointcept/datasets/generic_gs.py), which loads Gaussian attributes. We also implement global and local augmentations for different views.
* **Hooks**: We skip the testing stage by setting `evaluate=False`.


**Train from scratch.** Similar to the vision-language pretraining, you can start pretraining by calling the [pretrain](tools/ssl_pretrain.py) interface function with the corresponding configs.

Note that the config settings can be passed on-the-fly using the `--options` flag. The `save_path` is the directory where the model checkpoints, logs, and evaluation results will be saved. For example, the following command starts the run with the joint training config.
```bash
python tools/train.py \
    --config-file configs/concat_dataset/snellius/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py \
    --options save_path=exp_runs/lang_pretrainer/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive-new \
    batch_size=$batch_size batch_size_val=$batch_size_val \
    batch_size_test=$batch_size_test num_worker=$num_worker gpu_nums=$gpu_num \
    --num-gpus $gpu_num \
```

### Different Pretraining Mechanisms
The following options can be configured within the model class of the [config file](configs/concat_dataset/ssl-pretrain-concat-scan-ppv2-matt-3rscan-arkit-hyper-mcmc-base.py). To train only the student model using the masked loss, set `do_ema=False`, `do_ibot=False`, and `enable_mae_loss=True`.

If you'd like to use the **SimDINO** loss—which encourages the model to learn similar representations between global and local views—set `do_ema=False`.

To train the model to produce similar **dense features** between masked and unmasked views, set `do_ibot=True`.

For more details, refer to the [SimDINO repository](https://github.com/RobinWu218/SimDINO). Note that the SimDINO loss decreases slowly and thus requires a large batch size and more training time.

When you want to use the pretrained model for supervised training, first convert its checkpoints using the [model extraction script](scripts/rename_ckpt_dino_to_ptv3.py). You can then load the weights during supervised semantic segmentation training.

```
python tools/train.py \
    --config-file configs/scannet/semseg-gs-scannet-all-w-fix-xyz.py \
    --options save_path=exp_runs/scannet/<Experiment Name>\
    batch_size=$batch_size batch_size_val=$batch_size_val \
    batch_size_test=$batch_size_test num_worker=$num_worker gpu_nums=$gpu_num \
    weight=<model extract path>
    --num-gpus $gpu_num \
```

## Working with Custom Data
We cover using custom 3DGS scenes for vision-language feature inference.

**Inference on custom 3DGS scenes.** Please use the example script `scripts/preprocess_gs.py` to preprocess the 3DGS scenes into `*.npy` files. The inference requires the per-scene folder under a root path.

If there is no evaluation needed:
- Set `skip_eval=True` and `save_feat=True` in the tester settings
- Remove `class_names`, `text_embeddings` and `excluded_classes` settings
- Change the dataset type to `GenericGSDataset`
- Adjust the `split` and `data_root` path in the `data['test']` config

If to run zero-shot semantic segmentation on custom scenes:
- Obtain `segment.npy` during preprocessing (semseg labels can come from original dataset labels by neighboring voting, or be lifted with a custom pipeline)
- Make sure `segment.npy` has the same shape as other `*.npy` files, and use a consistent `ignore_index`
- Please refer to this [script](pointcept/datasets/preprocessing/holicity/preprocess_holicity_gs.py) for example, which preprocesses the HoliCity 3DGS scenes and obtains the `segment.npy` files
- Change the dataset type to `GenericGSDataset`
- Encode the class labels into text embeddings using the script `scripts/encode_labels.py`
- Set the corresponding `class_names`, `text_embeddings`, and `excluded_classes` settings in the tester config

Then load the model checkpoint and run the testing process individually.

## License

The 3D Gaussian Splatting scenes we provide are governed by the original dataset licenses as detailed in our data repository, please refer to the dataset license section [here](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k/blob/main/README.md#dataset-license). Our code, processing scripts, and metadata are made available under CC BY-SA 4.0.

## Citation
If you find our method/dataset helpful, please consider citing the following and/or ⭐ our repo.
```bib
@article{li2025scenesplat,
  title={Scenesplat: Gaussian splatting-based scene understanding with vision-language pretraining},
  author={Li, Yue and Ma, Qi and Yang, Runyi and Li, Huapeng and Ma, Mengjiao and Ren, Bin and Popovic, Nikola and Sebe, Nicu and Konukoglu, Ender and Gevers, Theo and others},
  journal={arXiv preprint arXiv:2503.18052},
  year={2025}
}
```

## Acknowledgement
We sincerely thank all the author teams of the original datasets for their contributions. Our work builds on the following repositories:

- [Pointcept](https://github.com/Pointcept/Pointcept) repository, on which we develop our codebase,
- [gsplat](https://github.com/nerfstudio-project/gsplat) repository, which we adapted to optimize the 3DGS scenes,
- [Occam's LGS](https://github.com/insait-institute/OccamLGS) repository, which we adapted for 3DGS pseudo label collection.

We are grateful to the authors for their open-source contributions!
