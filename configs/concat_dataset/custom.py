_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

# misc custom setting
debug = 1
gpu_nums = 1 if debug else 4
batch_size = 2 * gpu_nums
batch_size_val = 1 * gpu_nums
batch_size_test = 1 * gpu_nums
num_worker = 8 * gpu_nums if not debug else 1
mix_prob = 0.8
empty_cache = False
enable_amp = True
test_only = True
clip_grad = 1.0

# model settings
model = dict(
    type="LangPretrainer",
    backbone=dict(
        type="PT-v3m1",
        in_channels=11,  # gaussian: color 3, quaternion 4, scale 3, opacity 1, w/o normal 3
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2),
        enc_depths=(2, 2, 2, 6),
        enc_channels=(32, 64, 128, 256),  # -> this direction
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2),
        dec_channels=(768, 512, 256),  # <- this direction
        dec_num_head=(16, 16, 16),
        dec_patch_size=(1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CosineSimilarity", reduction="mean", loss_weight=1.0),
        dict(type="L2Loss", reduction="mean", loss_weight=1.0),
        dict(
            type="AggregatedContrastiveLoss",
            temperature=0.2,
            reduction="mean",
            loss_weight=0.02,
            schedule="last_75",
        ),
    ],
)

# scheduler settings
epoch = 800
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset root
repo_root = "/mnt/Data2/liyan/SceneSplat/"
scannet_data_root = "/home/yli7/scratch/datasets/gaussian_world/preprocessed/scannet_3dgs_mcmc_preprocessed"
scannetpp_data_root = (
    "/home/yli7/scratch/datasets/gaussian_world/preprocessed/scannetpp_v2_mcmc_3dgs"
)
matterport3d_data_root = "/home/yli7/scratch/datasets/gaussian_world/preprocessed/matterport3d_region_mcmc_3dgs"
holicity_data_root = (
    "/home/yli7/scratch/datasets/gaussian_world/preprocessed/holicity_mcmc_3dgs"
)
# matterport3d_data_root = "/mnt/Data3/liyan/preprocess/MP3D/"
customer_data_root = "/mnt/Data3/liyan/preprocess/MP3D/"

# training settings
feat_keys = ("color", "opacity", "quat", "scale")
grid_sample_keys = (
    "coord",
    "color",
    "opacity",
    "quat",
    "scale",
    "segment",
    "lang_feat",
    "valid_feat_mask",
)
grid_sample_keys_test = (
    "coord",
    "color",
    "opacity",
    "quat",
    "scale",
    "segment",
    "valid_feat_mask",
)
weight_pdnorm = {
    "Matterport3DGS": 3,
    "ScanNetPPGS": 2,
    "ScanNetGS": 2,
}  # MA3D as the main dataset

# trainer
train = dict(
    type="MultiDatasetTrainer",
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="LangPretrainZeroShotSemSegEvalMulti",  # ScanNet++, ScanNet200, Matterport3D_160
        class_names=[
            f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100.txt",
            f"{repo_root}/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_labels.txt",
            f"{repo_root}/pointcept/datasets/preprocessing/matterport3d/meta_data/matterport_nyu160_labels.txt",
        ],
        text_embeddings=[
            f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2.pt",
            f"{repo_root}/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_text_embeddings_siglip2.pt",
            f"{repo_root}/pointcept/datasets/preprocessing/matterport3d/meta_data/matterport-nyu160_text_embeddings_siglip2.pt",
        ],
        excluded_classes=[
            ["wall", "floor", "ceiling"],
            ["wall", "floor", "ceiling"],
            ["wall", "floor", "ceiling", "other furniture"],
        ],
        pred_label_mapping=[None, None, None],
        ignore_index=-1,
        vote_k=25,
        enable_voting=True,
        confidence_threshold=0.1,
    ),
    dict(type="CheckpointSaver", save_freq=None),
    dict(
        type="PreciseEvaluator", test_last=False if not test_only else True
    ),  # use test_last=True to use current / loaded weight for evaluation
]

data = dict(
    num_classes=100,
    ignore_index=-1,
    train=dict(
        type="ConcatDataset",
        datasets=[
            # matterport3d
            dict(
                type="Matterport3D_160_GSDataset",
                split=(
                    "train_grid1.0cm_chunk6x6_stride3x3_filtered",
                    "val_grid1.0cm_chunk6x6_stride3x3_filtered",
                ),
                data_root=matterport3d_data_root,
                sample_tail_classes=False,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2,
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.01),
                    dict(
                        type="ElasticDistortion",
                        distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    ),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=grid_sample_keys,
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", point_max=192000, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    # dict(type="NormalizeCoord"),
                    # dict(type="ShufflePoint"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=(
                            "coord",
                            "grid_coord",
                            "segment",
                            "lang_feat",
                            "valid_feat_mask",
                            "pc_coord",
                            "pc_segment",
                        ),
                        feat_keys=feat_keys,
                    ),
                ],
                test_mode=False,
                loop=weight_pdnorm["Matterport3DGS"],
            ),
            # scannet++
            dict(
                type="ScanNetPPGSDataset",
                split=(
                    "train_grid1.0cm_chunk6x6_stride3x3",
                    "test_grid1.0cm_chunk6x6_stride3x3",
                ),
                data_root=scannetpp_data_root,
                sample_tail_classes=False,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2,
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.01),
                    dict(
                        type="ElasticDistortion",
                        distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    ),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=grid_sample_keys,
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", point_max=192000, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    # dict(type="NormalizeCoord"),
                    # dict(type="ShufflePoint"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=(
                            "coord",
                            "grid_coord",
                            "segment",
                            "lang_feat",
                            "valid_feat_mask",
                            "pc_coord",
                            "pc_segment",
                        ),
                        feat_keys=feat_keys,
                    ),
                ],
                test_mode=False,
                loop=weight_pdnorm["ScanNetPPGS"],
            ),
            # scannet
            dict(
                type="ScanNet200GSDataset",
                split=(
                    "train_grid1.0cm_chunk6x6_stride3x3",
                    "test_grid1.0cm_chunk6x6_stride3x3",
                ),
                data_root=scannet_data_root,
                sample_tail_classes=False,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2,
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.01),
                    dict(
                        type="ElasticDistortion",
                        distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    ),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=grid_sample_keys,
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", point_max=192000, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    # dict(type="NormalizeCoord"),
                    # dict(type="ShufflePoint"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=(
                            "coord",
                            "grid_coord",
                            "segment",
                            "lang_feat",
                            "valid_feat_mask",
                            "pc_coord",
                            "pc_segment",
                        ),
                        feat_keys=feat_keys,
                    ),
                ],
                test_mode=False,
                loop=weight_pdnorm["ScanNetGS"],
            ),
        ],
    ),
    val=[
        # scannet++
        dict(
            type="ScanNetPPGSDataset",
            split=("val"),
            data_root=scannetpp_data_root,
            is_train=False,  # if not in train, we will load pc_segment and pc_coord if exsit, and not downsample them
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    keys=grid_sample_keys,
                    return_grid_coord=True,
                ),
                # dict(type="SphereCrop", point_max=600000, mode="random"), # spconv limitation: int64_t(N) * int64_t(C) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max, i.e., max 698k points for inference
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=(
                        "coord",
                        "grid_coord",
                        "segment",
                        "lang_feat",
                        "valid_feat_mask",
                        "pc_coord",
                        "pc_segment",
                    ),
                    feat_keys=feat_keys,
                ),
            ],
            test_mode=False,
        ),
        # scannet200
        dict(
            type="ScanNet200GSDataset",
            split=("val"),
            data_root=scannet_data_root,
            is_train=False,  # if not in train, we will load pc_segment and pc_coord if exsit, and not downsample them
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    keys=grid_sample_keys,
                    return_grid_coord=True,
                ),
                # dict(type="SphereCrop", point_max=600000, mode="random"), # spconv limitation: int64_t(N) * int64_t(C) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max, i.e., max 698k points for inference
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=(
                        "coord",
                        "grid_coord",
                        "segment",
                        "lang_feat",
                        "valid_feat_mask",
                        "pc_coord",
                        "pc_segment",
                    ),
                    feat_keys=feat_keys,
                ),
            ],
            test_mode=False,
        ),
        # matterport3d_160
        dict(
            type="Matterport3D_160_GSDataset",
            split=("test_grid1.0cm_chunk6x6_stride3x3_filtered"),
            data_root=matterport3d_data_root,
            is_train=False,  # if not in train, we will load pc_segment and pc_coord if exsit
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    keys=grid_sample_keys,
                    return_grid_coord=True,
                ),
                # dict(type="SphereCrop", point_max=600000, mode="random"), # spconv limitation: int64_t(N) * int64_t(C) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max, i.e., max 698k points for inference
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=(
                        "coord",
                        "grid_coord",
                        "segment",
                        "lang_feat",
                        "valid_feat_mask",
                        "pc_coord",
                        "pc_segment",
                    ),
                    feat_keys=feat_keys,
                ),
            ],
            test_mode=False,
        ),
    ],
    test=[
        # scannet20
        # dict(
        #     type="ScanNetGSDataset",
        #     split="val",
        #     data_root=scannet_data_root,
        #     is_train=False,
        #     transform=[
        #         dict(type="CenterShift", apply_z=True),
        #         dict(type="NormalizeColor"),
        #         dict(
        #             type="Copy",
        #             keys_dict=dict(
        #                 segment="origin_segment",
        #                 coord="origin_coord",
        #                 valid_feat_mask="origin_feat_mask",
        #                 pc_instance="origin_instance",
        #             ),
        #         ),
        #         dict(
        #             type="GridSample",
        #             grid_size=0.01,
        #             hash_type="fnv",
        #             mode="train",
        #             keys=grid_sample_keys,
        #             apply_to_pc=False,
        #             return_inverse=True,
        #         ),
        #     ],
        #     test_mode=True,
        #     test_cfg=dict(
        #         voxelize=dict(
        #             type="GridSample",
        #             grid_size=0.02,
        #             hash_type="fnv",
        #             mode="test",
        #             keys=grid_sample_keys_test,  # keep keys for inference is enough here
        #             apply_to_pc=False,
        #             return_grid_coord=True,
        #         ),
        #         crop=None,
        #         post_transform=[
        #             dict(type="CenterShift", apply_z=False),
        #             dict(type="ToTensor"),
        #             dict(
        #                 type="Collect",
        #                 keys=(
        #                     "coord",
        #                     "grid_coord",
        #                     "index",
        #                     "lang_feat",
        #                     "valid_feat_mask",
        #                     "pc_coord",
        #                     "pc_segment",
        #                 ),
        #                 feat_keys=feat_keys,
        #             ),  # only keys for inference
        #         ],
        #         aug_transform=[
        #             [
        #                 {
        #                     "type": "RandomRotateTargetAngle",
        #                     "angle": [0],
        #                     "axis": "z",
        #                     "center": [0, 0, 0],
        #                     "p": 1,
        #                 }
        #             ]
        #         ],
        #     ),
        # ),
        # # scannet200
        # dict(
        #     type="ScanNet200GSDataset",
        #     split="val",
        #     data_root=scannet_data_root,
        #     is_train=False,
        #     transform=[
        #         dict(type="CenterShift", apply_z=True),
        #         dict(type="NormalizeColor"),
        #         dict(
        #             type="Copy",
        #             keys_dict=dict(
        #                 segment="origin_segment",
        #                 coord="origin_coord",
        #                 valid_feat_mask="origin_feat_mask",
        #                 pc_instance="origin_instance",
        #             ),
        #         ),
        #         dict(
        #             type="GridSample",
        #             grid_size=0.01,
        #             hash_type="fnv",
        #             mode="train",
        #             keys=grid_sample_keys,
        #             apply_to_pc=False,
        #             return_inverse=True,
        #         ),
        #     ],
        #     test_mode=True,
        #     test_cfg=dict(
        #         voxelize=dict(
        #             type="GridSample",
        #             grid_size=0.02,
        #             hash_type="fnv",
        #             mode="test",
        #             keys=grid_sample_keys_test,  # keep keys for inference is enough here
        #             apply_to_pc=False,
        #             return_grid_coord=True,
        #         ),
        #         crop=None,
        #         post_transform=[
        #             dict(type="CenterShift", apply_z=False),
        #             dict(type="ToTensor"),
        #             dict(
        #                 type="Collect",
        #                 keys=(
        #                     "coord",
        #                     "grid_coord",
        #                     "index",
        #                     "lang_feat",
        #                     "valid_feat_mask",
        #                     "pc_coord",
        #                     "pc_segment",
        #                 ),
        #                 feat_keys=feat_keys,
        #             ),  # only keys for inference
        #         ],
        #         aug_transform=[
        #             [
        #                 {
        #                     "type": "RandomRotateTargetAngle",
        #                     "angle": [0],
        #                     "axis": "z",
        #                     "center": [0, 0, 0],
        #                     "p": 1,
        #                 }
        #             ]
        #         ],
        #     ),
        # ),
        # # matterport3d_160
        # dict(
        #     type="Matterport3D_160_GSDataset",
        #     split="test",
        #     data_root=matterport3d_data_root,
        #     is_train=False,
        #     transform=[
        #         dict(type="CenterShift", apply_z=True),
        #         dict(type="NormalizeColor"),
        #         dict(
        #             type="Copy",
        #             keys_dict=dict(
        #                 segment="origin_segment",
        #                 coord="origin_coord",
        #                 valid_feat_mask="origin_feat_mask",
        #             ),
        #         ),
        #         dict(
        #             type="GridSample",
        #             grid_size=0.01,
        #             hash_type="fnv",
        #             mode="train",
        #             keys=grid_sample_keys,
        #             apply_to_pc=False,
        #             return_inverse=True,
        #         ),
        #     ],
        #     test_mode=True,
        #     test_cfg=dict(
        #         voxelize=dict(
        #             type="GridSample",
        #             grid_size=0.02,
        #             hash_type="fnv",
        #             mode="test",
        #             keys=grid_sample_keys_test,  # keep keys for inference is enough here
        #             apply_to_pc=False,
        #             return_grid_coord=True,
        #         ),
        #         crop=None,
        #         post_transform=[
        #             dict(type="CenterShift", apply_z=False),
        #             dict(type="ToTensor"),
        #             dict(
        #                 type="Collect",
        #                 keys=(
        #                     "coord",
        #                     "grid_coord",
        #                     "index",
        #                     "lang_feat",
        #                     "valid_feat_mask",
        #                     "pc_coord",
        #                     "pc_segment",
        #                 ),
        #                 feat_keys=feat_keys,
        #             ),  # only keys for inference
        #         ],
        #         aug_transform=[
        #             [
        #                 {
        #                     "type": "RandomRotateTargetAngle",
        #                     "angle": [0],
        #                     "axis": "z",
        #                     "center": [0, 0, 0],
        #                     "p": 1,
        #                 }
        #             ]
        #         ],
        #     ),
        # ),
        # # scannet++
        # dict(
        #     type="ScanNetPPGSDataset",
        #     split="val",
        #     data_root=scannetpp_data_root,
        #     is_train=False,
        #     transform=[
        #         dict(type="CenterShift", apply_z=True),
        #         dict(type="NormalizeColor"),
        #         dict(
        #             type="Copy",
        #             keys_dict=dict(
        #                 segment="origin_segment",
        #                 coord="origin_coord",
        #                 valid_feat_mask="origin_feat_mask",
        #             ),
        #         ),
        #         dict(
        #             type="GridSample",
        #             grid_size=0.01,
        #             hash_type="fnv",
        #             mode="train",
        #             keys=grid_sample_keys,
        #             apply_to_pc=False,
        #             return_inverse=True,
        #         ),
        #     ],
        #     test_mode=True,
        #     test_cfg=dict(
        #         voxelize=dict(
        #             type="GridSample",
        #             grid_size=0.02,
        #             hash_type="fnv",
        #             mode="test",
        #             keys=grid_sample_keys_test,  # keep keys for inference is enough here
        #             apply_to_pc=False,
        #             return_grid_coord=True,
        #         ),
        #         crop=None,
        #         post_transform=[
        #             dict(type="CenterShift", apply_z=False),
        #             dict(type="ToTensor"),
        #             dict(
        #                 type="Collect",
        #                 keys=(
        #                     "coord",
        #                     "grid_coord",
        #                     "index",
        #                     "lang_feat",
        #                     "valid_feat_mask",
        #                     "pc_coord",
        #                     "pc_segment",
        #                 ),
        #                 feat_keys=feat_keys,
        #             ),  # only keys for inference
        #         ],
        #         aug_transform=[
        #             [
        #                 {
        #                     "type": "RandomRotateTargetAngle",
        #                     "angle": [0],
        #                     "axis": "z",
        #                     "center": [0, 0, 0],
        #                     "p": 1,
        #                 }
        #             ]
        #         ],
        #     ),
        # ),
        #  customer
        dict(
            type="GenericGSDataset",
            split="SceneSplat",
            data_root=customer_data_root,
            is_train=False,
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="NormalizeColor"),
                dict(
                    type="Copy",
                    keys_dict=dict(
                        segment="origin_segment",
                        coord="origin_coord",
                        valid_feat_mask="origin_feat_mask",
                        pc_instance="origin_instance",
                    ),
                ),
                dict(
                    type="GridSample",
                    grid_size=0.01,
                    hash_type="fnv",
                    mode="train",
                    keys=grid_sample_keys,
                    apply_to_pc=False,
                    return_inverse=True,
                ),
            ],
            test_mode=True,
            test_cfg=dict(
                voxelize=dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="test",
                    keys={ "coord",
                        "color",
                        "opacity",
                        "quat",
                        "scale",
                        "lang_feat",
                        "valid_feat_mask",},  # keep keys for inference is enough here
                    apply_to_pc=False,
                    return_grid_coord=True,
                    return_inverse=True,
                ),
                crop=None,
                post_transform=[
                    dict(type="CenterShift", apply_z=False),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=(
                            "coord",
                            "grid_coord",
                            "index",
                            "lang_feat",
                            "valid_feat_mask",
                            "pc_coord",
                            "pc_segment",
                        ),
                        feat_keys=("color", "opacity", "quat", "scale"),
                    ),  # only keys for inference
                ],
                aug_transform=[
                    [
                        {
                            "type": "RandomRotateTargetAngle",
                            "angle": [0],
                            "axis": "z",
                            "center": [0, 0, 0],
                            "p": 1,
                        }
                    ]
                ],
            ),
        ),



        ###########################
    ],
)

# Tester
test = [
    # scannet20
    # dict(
    #     type="ZeroShotSemSegTester",
    #     verbose=True,
    #     class_names=f"{repo_root}/pointcept/datasets/preprocessing/scannet/meta_data/scannet20_labels.txt",
    #     text_embeddings=f"{repo_root}/pointcept/datasets/preprocessing/scannet/meta_data/scannet20_text_embeddings_siglip2.pt",
    #     excluded_classes=["wall", "floor", "ceiling"],
    #     enable_voting=True,
    #     vote_k=25,
    #     confidence_threshold=0.1,
    #     save_feat=False,
    #     skip_eval=False,
    # ),
    # scannet200
    # dict(
    #     type="ZeroShotSemSegTester",
    #     verbose=True,
    #     class_names=f"{repo_root}/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_labels.txt",
    #     text_embeddings=f"{repo_root}/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_text_embeddings_siglip2.pt",
    #     excluded_classes=["wall", "floor", "ceiling"],
    #     enable_voting=True,
    #     vote_k=25,
    #     confidence_threshold=0.1,
    #     save_feat=False,
    #     skip_eval=False,
    # ),
    # matterport3d_160
    # dict(
    #     type="ZeroShotSemSegTester",
    #     verbose=True,
    #     class_names=f"{repo_root}/pointcept/datasets/preprocessing/matterport3d/meta_data/matterport_nyu160_labels.txt",
    #     text_embeddings=f"{repo_root}/pointcept/datasets/preprocessing/matterport3d/meta_data/matterport-nyu160_text_embeddings_siglip2.pt",
    #     excluded_classes=["wall", "floor", "ceiling", "other furniture"],
    #     enable_voting=True,
    #     vote_k=25,
    #     confidence_threshold=0.1,
    #     save_feat=False,
    #     skip_eval=False,
    # ),
    # scannet++
    # dict(
    #     type="ZeroShotSemSegTester",
    #     verbose=True,
    #     class_names=f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100.txt",
    #     text_embeddings=f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2.pt",
    #     excluded_classes=["wall", "floor", "ceiling"],
    #     enable_voting=True,
    #     vote_k=25,
    #     confidence_threshold=0.1,
    #     save_feat=False,
    #     skip_eval=False,
    # ),
    # customer
    dict(
        type="ZeroShotSemSegTester",
        verbose=True,
        class_names=f"{repo_root}/pointcept/datasets/preprocessing/matterport3d/meta_data/mp3d40.txt",
        text_embeddings=f"{repo_root}/pointcept/datasets/preprocessing/matterport3d/meta_data/mp3d40_text_embeddings_siglip2.pt",
        excluded_classes=["wall", "floor", "ceiling", "other furniture"],
        enable_voting=True,
        vote_k=25,
        confidence_threshold=0.1,
        save_feat=True,
        skip_eval=False,
    ),
]
