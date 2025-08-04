import os
import argparse
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path


def chunking_scene(
    name,
    dataset_root,
    output_dir,
    split,
    grid_size=None,
    chunk_range=(6, 6),
    chunk_stride=(3, 3),
    chunk_minimum_size=10000,
    max_chunk_num=None,
    debug=False,
):
    if debug:
        print("=============================")
        print("DEBUG MODE, turn off to save chunks")
    print(f"Chunking scene {name} in {split} split")
    dataset_root = Path(dataset_root)
    scene_path = dataset_root / split / name
    assets = os.listdir(scene_path)
    data_dict = dict()
    for asset in assets:
        if not asset.endswith(".npy"):
            continue
        data_dict[asset[:-4]] = np.load(scene_path / asset)
    # recenter the coordinates!!!
    coord = data_dict["coord"] - data_dict["coord"].min(axis=0)

    if "lang_feat" in data_dict.keys():
        valid_feat_mask = data_dict["valid_feat_mask"]
        # normalize the valid part in lang_feat
        data_dict["lang_feat"][valid_feat_mask == 1] /= np.linalg.norm(
            data_dict["lang_feat"][valid_feat_mask == 1], axis=1, keepdims=True
        )
    if debug:
        coord_length = len(coord)
    if grid_size is not None:
        grid_coord = np.floor(coord / grid_size).astype(int)

        # Group indices by grid cell (each grid cell is represented as a tuple)
        grid_to_indices = {}
        for i, cell in enumerate(grid_coord):
            cell_tuple = tuple(cell)
            grid_to_indices.setdefault(cell_tuple, []).append(i)

        selected_idx = []
        for indices in grid_to_indices.values():
            # If 'valid_feat_mask' is in data_dict, try to pick one with a valid mask of 1.
            if "valid_feat_mask" in data_dict:
                valid_indices = [
                    i for i in indices if data_dict["valid_feat_mask"][i] == 1
                ]
                if valid_indices:
                    chosen = np.random.choice(valid_indices)
                    # chosen = valid_indices[0]
                else:
                    # No valid entries; fall back to one of the indices (here, the first one)
                    chosen = indices[0]
            else:
                # If there is no valid_feat_mask, just pick the first entry as before using np.unique().
                chosen = indices[0]
            selected_idx.append(chosen)

        selected_idx = np.array(selected_idx)
        coord = coord[selected_idx]
        for key in data_dict.keys():
            data_dict[key] = data_dict[key][selected_idx]

    bev_range = coord.max(axis=0)[:2]
    bev_start = coord.min(axis=0)[:2]  # debug
    # debug
    if debug:
        print("name", name)
        print("bev_range", bev_range)  # max in x, y
        print("bev_start", bev_start)  # min in x, y
    x, y = np.meshgrid(
        # note small rooms will be skipped here!
        # e.g, 2.5+3-6<0, there will be no chunk in this case
        np.arange(0, bev_range[0] + chunk_stride[0] - chunk_range[0], chunk_stride[0]),
        # np.arange(0, bev_range[0] + chunk_stride[0] - chunk_range[0], chunk_stride[0]),
        np.arange(0, bev_range[1] + chunk_stride[1] - chunk_range[1], chunk_stride[1]),
        indexing="ij",
    )
    chunks = np.concatenate([x.reshape([-1, 1]), y.reshape([-1, 1])], axis=-1)
    chunk_idx = 0

    if max_chunk_num is not None and len(chunks) > max_chunk_num:
        # selecte a subset of chunks with most points
        chunk_points = []
        for chunk in chunks:
            mask = (
                (coord[:, 0] >= chunk[0])
                & (coord[:, 0] < chunk[0] + chunk_range[0])
                & (coord[:, 1] >= chunk[1])
                & (coord[:, 1] < chunk[1] + chunk_range[1])
            )

            chunk_points.append(np.sum(mask))
        chunk_points = np.array(chunk_points)
        chunk_points = np.argsort(chunk_points)[::-1]
        chunks = chunks[chunk_points[:max_chunk_num]]
        print(f"selected {max_chunk_num} chunks with most points for {name}")

    for chunk in chunks:
        if debug:
            print("name", name, "chunk", chunk, "chunk_range", chunk_range)
        mask = (
            (coord[:, 0] >= chunk[0])
            & (coord[:, 0] < chunk[0] + chunk_range[0])
            & (coord[:, 1] >= chunk[1])
            & (coord[:, 1] < chunk[1] + chunk_range[1])
        )

        if np.sum(mask) < chunk_minimum_size:
            print(
                f"{name} Chunk size too small, skip,",
                "coords within the chunk: ",
                np.sum(mask),
            )
            continue

        chunk_data_name = f"{name}_{chunk_idx}"
        if grid_size is not None:
            chunk_split_name = (
                f"{split}_"
                f"grid{grid_size * 100:.1f}cm_"
                f"chunk{chunk_range[0]}x{chunk_range[1]}_"
                f"stride{chunk_stride[0]}x{chunk_stride[1]}"
            )
        else:
            chunk_split_name = (
                f"{split}_"
                f"chunk{chunk_range[0]}x{chunk_range[1]}_"
                f"stride{chunk_stride[0]}x{chunk_stride[1]}"
            )

        if output_dir is not None:
            chunk_save_path = Path(output_dir) / chunk_split_name / chunk_data_name
        else:
            chunk_save_path = dataset_root / chunk_split_name / chunk_data_name
        chunk_save_path.mkdir(parents=True, exist_ok=True)
        for key in data_dict.keys():
            np.save(chunk_save_path / f"{key}.npy", data_dict[key][mask])

        chunk_idx += 1

    print(f"in total {chunk_idx} valid chunks in {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the Pointcept processed ScanNet++ dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Path to the Pointcept processed ScanNet++ dataset.",
    )
    parser.add_argument(
        "--split",
        required=True,
        default="train",
        type=str,
        help="Split need to process.",
    )
    parser.add_argument(
        "--grid_size",
        default=None,
        type=float,
        help="Grid size for initial grid sampling",
    )
    parser.add_argument(
        "--chunk_range",
        default=[6, 6],
        type=int,
        nargs="+",
        help="Range of each chunk, e.g. --chunk_range 6 6",
    )
    parser.add_argument(
        "--chunk_stride",
        default=[3, 3],
        type=int,
        nargs="+",
        help="Stride of each chunk, e.g. --chunk_stride 3 3",
    )
    parser.add_argument(
        "--chunk_minimum_size",
        default=10000,
        type=int,
        help="Minimum number of points in each chunk",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        "--subset_list",
        type=str,
        default=None,
        help=(
            "which dataset entries (folders or files) to keep. "
            "If omitted, all entries are used."
        ),
    )
    parser.add_argument(
        "--max_chunk_num",
        type=int,
        default=None,
        help="Maximum number of chunks to process.",
    )
    parser.add_argument("--single_process", action="store_true")

    config = parser.parse_args()
    config.dataset_root = Path(config.dataset_root)
    data_list = os.listdir(config.dataset_root / config.split)
    if config.subset_list is not None:
        subset_path = Path(config.subset_list)
        with subset_path.open("r") as f:
            subset_list = {line.strip() for line in f if line.strip()}
        data_list = [name for name in data_list if name in subset_list]

    print(f"Processing {len(data_list)} scenes in {config.split} split")
    if not config.single_process:
        pool = ProcessPoolExecutor(max_workers=config.num_workers)
        _ = list(
            pool.map(
                chunking_scene,
                data_list,
                repeat(config.dataset_root),
                repeat(config.output_dir),
                repeat(config.split),
                repeat(config.grid_size),
                repeat(config.chunk_range),
                repeat(config.chunk_stride),
                repeat(config.chunk_minimum_size),
                repeat(config.max_chunk_num),
            )
        )
        pool.shutdown()
    else:
        for name in data_list:
            chunking_scene(
                name,
                config.dataset_root,
                config.output_dir,
                config.split,
                config.grid_size,
                config.chunk_range,
                config.chunk_stride,
                config.chunk_minimum_size,
                config.max_chunk_num,
            )
    print("All scenes chunked!")
