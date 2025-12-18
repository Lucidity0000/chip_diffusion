"""
Visualize placement pickle files (e.g. SA-hybrid `best*.pkl`) as PNGs.

Examples (PowerShell):
  $env:PYTHONPATH='.'
  python scripts/visualize_placement_pkls.py --task vertex_0.7x.61 --pkl logs/vertex_0.7x.61.sa_hybrid.0/samples/best0.pkl

  python scripts/visualize_placement_pkls.py --task vertex_0.7x.61 --pkl-dir logs/vertex_0.7x.61.sa_hybrid.0/samples --pattern "best*.pkl" --save-ref
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from PIL import Image
from torch_geometric.data import Data

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIFFUSION_DIR = _REPO_ROOT / "diffusion"
# Many modules in this repo use non-package imports like `import orientations`,
# which rely on the `diffusion/` directory being on `sys.path`.
for _path in (str(_REPO_ROOT), str(_DIFFUSION_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from diffusion import utils  # noqa: E402


def _extract_last_int(text: str) -> Optional[int]:
    matches = re.findall(r"(\d+)", text)
    return int(matches[-1]) if matches else None


def _to_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, torch.Tensor):
        return int(value.item())
    try:
        return int(value)
    except Exception:
        return None


def _load_dataset(task: str, split: str):
    train_set, val_set = utils.load_graph_data_with_config(task, train_data_limit="none", val_data_limit="none")
    if split == "train":
        return train_set
    if split == "val":
        return val_set
    # auto
    return val_set if len(val_set) > 0 else train_set


def _dataset_has_file_idx(dataset) -> bool:
    for x_ref, cond in dataset[: min(5, len(dataset))]:
        _ = x_ref
        if hasattr(cond, "file_idx") or ("file_idx" in cond):
            return True
    return False


def _select_sample(dataset, sample_id: int, index_mode: str):
    if index_mode == "dataset_index":
        if sample_id < 0 or sample_id >= len(dataset):
            raise IndexError(f"dataset index {sample_id} out of range (size={len(dataset)})")
        return dataset[sample_id]

    if index_mode == "file_idx":
        for x_ref, cond in dataset:
            file_idx = _to_int(getattr(cond, "file_idx", None))
            if file_idx == sample_id:
                return x_ref, cond
        raise KeyError(f"no sample with cond.file_idx == {sample_id} found in dataset (size={len(dataset)})")

    # auto
    if _dataset_has_file_idx(dataset):
        try:
            return _select_sample(dataset, sample_id, "file_idx")
        except KeyError:
            return _select_sample(dataset, sample_id, "dataset_index")
    return _select_sample(dataset, sample_id, "dataset_index")


def _get_dataset_scale(task: str) -> float:
    try:
        cfg = utils.get_dataset_config(task)
        if "scale" in cfg:
            return float(cfg.scale)
    except Exception:
        pass
    return 1.0


def _chip_size_and_offset(cond: Data) -> Tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(cond, "chip_size") and ("chip_size" not in cond):
        raise ValueError("cond has no chip_size; cannot convert from physical coordinates")
    chip_size = cond.chip_size
    chip_size = torch.tensor(chip_size, dtype=torch.float32) if not isinstance(chip_size, torch.Tensor) else chip_size.float()
    if chip_size.numel() == 2:
        size = chip_size.view(2)
        offset = torch.zeros(2, dtype=torch.float32)
    elif chip_size.numel() == 4:
        size = (chip_size[2:] - chip_size[:2]).view(2)
        offset = chip_size[:2].view(2)
    else:
        raise ValueError(f"unexpected chip_size shape: {tuple(chip_size.shape)}")
    return size, offset


def _physical_bottom_left_to_normalized_center(
    x_bottom_left: torch.Tensor, cond: Data, dataset_scale: float
) -> torch.Tensor:
    """
    Convert physical bottom-left coords (as saved by diffusion/eval.py for benchmarks) into normalized center coords
    expected by diffusion.utils.visualize_placement.
    """
    chip_size, chip_offset = _chip_size_and_offset(cond)
    chip_size = chip_size.view(1, 2)
    chip_offset = chip_offset.view(1, 2)
    x = x_bottom_left.to(dtype=torch.float32)
    x = (x - chip_offset) / float(dataset_scale)
    x = 2 * (x / chip_size) - 1
    x = x + (cond.x.to(dtype=torch.float32) / 2)
    return x


def _load_placement_from_pkl(pkl_path: Path) -> torch.Tensor:
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        obj = obj[0]
    if isinstance(obj, torch.Tensor):
        x = obj.detach().cpu().float()
    else:
        x = torch.tensor(obj, dtype=torch.float32)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError(f"expected placement shaped (V, 2+), got {tuple(x.shape)} from {pkl_path}")
    return x[:, :2].contiguous()


def _apply_filter(x: torch.Tensor, cond: Data, filter_mode: str) -> Tuple[torch.Tensor, Data]:
    if filter_mode == "all":
        return x, cond
    if not hasattr(cond, "is_macros") and ("is_macros" not in cond):
        raise ValueError(f"--filter {filter_mode} requires cond.is_macros, but dataset has no is_macros")
    is_macros = cond.is_macros.bool() if isinstance(cond.is_macros, torch.Tensor) else torch.tensor(cond.is_macros).bool()
    is_ports = cond.is_ports.bool() if hasattr(cond, "is_ports") else torch.zeros_like(is_macros)

    if filter_mode == "macros":
        keep = is_macros
    elif filter_mode == "macros_and_ports":
        keep = is_macros | is_ports
    else:
        raise ValueError(f"unknown filter mode: {filter_mode}")

    idx = torch.nonzero(keep).flatten()
    x_sub = x[idx]
    cond_sub = Data(
        x=cond.x[idx],
        is_ports=is_ports[idx],
        is_macros=is_macros[idx],
    )
    # preserve chip_size for coordinate conversion callers (if needed later)
    if hasattr(cond, "chip_size") or ("chip_size" in cond):
        cond_sub.chip_size = cond.chip_size
    return x_sub, cond_sub


def _save_png(img_array, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_array).save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize placement .pkl files as PNGs")
    parser.add_argument("--task", required=True, help="Dataset/task name (e.g. vertex_0.7x.61, ispd2005-s0)")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pkl", nargs="+", type=Path, help="One or more placement .pkl files to visualize")
    input_group.add_argument("--pkl-dir", type=Path, help="Directory containing placement .pkl files")
    parser.add_argument("--pattern", default="*.pkl", help="Glob pattern for --pkl-dir (default: *.pkl)")

    parser.add_argument(
        "--split",
        choices=["auto", "train", "val"],
        default="auto",
        help="Which dataset split to pull graphs from (default: auto)",
    )
    parser.add_argument(
        "--index-mode",
        choices=["auto", "file_idx", "dataset_index"],
        default="auto",
        help="How to map sample ids to dataset entries (default: auto)",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        default=None,
        help="Sample id to use when it cannot be inferred from the filename",
    )

    parser.add_argument(
        "--format",
        choices=["auto", "normalized_center", "physical_bottom_left"],
        default="auto",
        help="Placement coordinate format in the .pkl file (default: auto)",
    )
    parser.add_argument("--img-size", nargs=2, type=int, default=[1024, 1024], help="Image size W H (default: 1024 1024)")
    parser.add_argument("--plot-pins", action="store_true", help="Plot pin locations (can be slow)")
    parser.add_argument("--plot-edges", action="store_true", help="Plot net edges (can be slow)")
    parser.add_argument(
        "--filter",
        choices=["all", "macros", "macros_and_ports"],
        default="all",
        help="Limit visualization to a subset of nodes (default: all)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: alongside each .pkl)")
    parser.add_argument("--save-ref", action="store_true", help="Also save reference placement from the dataset")

    args = parser.parse_args()

    pkls: List[Path]
    if args.pkl is not None:
        pkls = [p.expanduser().resolve() for p in args.pkl]
    else:
        pkls = sorted([p.resolve() for p in args.pkl_dir.glob(args.pattern)])
        if not pkls:
            raise FileNotFoundError(f"no files matched {args.pkl_dir} / {args.pattern}")

    dataset = _load_dataset(args.task, args.split)
    if len(dataset) == 0:
        raise RuntimeError(f"no samples found for task={args.task} (split={args.split})")

    dataset_scale = _get_dataset_scale(args.task)
    img_size = (int(args.img_size[0]), int(args.img_size[1]))

    for pkl_path in pkls:
        inferred_id = _extract_last_int(pkl_path.stem)
        sample_id = args.sample_id if args.sample_id is not None else inferred_id
        if sample_id is None:
            raise ValueError(f"could not infer sample id from filename {pkl_path.name}; pass --sample-id")

        x_ref, cond = _select_sample(dataset, sample_id, args.index_mode)
        x_in = _load_placement_from_pkl(pkl_path)

        placement_format = args.format
        if placement_format == "auto":
            if (hasattr(cond, "chip_size") or ("chip_size" in cond)) and (x_in.max().item() > 2.5 or x_in.min().item() < -2.5):
                placement_format = "physical_bottom_left"
            else:
                placement_format = "normalized_center"

        if placement_format == "physical_bottom_left":
            x_vis = _physical_bottom_left_to_normalized_center(x_in, cond, dataset_scale)
        else:
            x_vis = x_in

        if args.filter != "all" and (args.plot_pins or args.plot_edges):
            raise ValueError("--plot-pins/--plot-edges are not supported with --filter != all (subgraph not built)")

        x_vis, cond_vis = _apply_filter(x_vis, cond, args.filter)
        img = utils.visualize_placement(x_vis, cond_vis, plot_pins=args.plot_pins, plot_edges=args.plot_edges, img_size=img_size)

        out_dir = args.out_dir if args.out_dir is not None else pkl_path.parent
        out_path = Path(out_dir) / f"{pkl_path.stem}.png"
        _save_png(img, out_path)
        print(f"wrote {out_path}")

        if args.save_ref:
            x_ref_vis, cond_ref_vis = _apply_filter(x_ref, cond, args.filter)
            img_ref = utils.visualize_placement(
                x_ref_vis, cond_ref_vis, plot_pins=args.plot_pins, plot_edges=args.plot_edges, img_size=img_size
            )
            ref_path = Path(out_dir) / f"{pkl_path.stem}_ref.png"
            _save_png(img_ref, ref_path)
            print(f"wrote {ref_path}")

    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", ".")
    raise SystemExit(main())
