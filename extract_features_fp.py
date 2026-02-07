#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
import h5py
import openslide
from tqdm import tqdm
import numpy as np
from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder  
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_slide_features_only(output_path, loader, model, verbose=0):
    """
        HDF5: features -> float32 [N, D], coords -> int32 [N, 2]
    """
    if verbose > 0:
        print(f'processing {len(loader)} batches (collect all patches first)')
    zs, coords_all = [], []
    n_batches = 0
    for _, data in enumerate(tqdm(loader)):
        n_batches += 1
        if ('img' not in data) or ('coord' not in data):
            raise RuntimeError(f"[E] dataloader batch missing keys. got keys={list(data.keys())}")

        imgs = data['img']
        coords_b = data['coord']

        if imgs is None or coords_b is None:
            if verbose > 0:
                print("[W] skip batch: imgs or coords is None")
            continue
        if getattr(imgs, "numel", lambda: 0)() == 0:
            if verbose > 0:
                print("[W] skip batch: imgs.numel()==0")
            continue

        imgs = imgs.to(device, non_blocking=True)               # [B,3,H,W]
        coords_np = np.asarray(coords_b, dtype=np.int32)        # [B,2]
        if coords_np.ndim != 2 or coords_np.shape[1] != 2:
            raise RuntimeError(f"[E] coords shape expected [B,2], got {coords_np.shape}")
        with torch.inference_mode():
            z = model(imgs)  
            if not isinstance(z, torch.Tensor):
                raise RuntimeError(f"[E] model(imgs) must return tensor, got {type(z)}")

            if z.ndim == 1:
                z = z.unsqueeze(1)
            elif z.ndim > 2:
                z = z.view(z.size(0), -1)
            if z.ndim != 2:
                raise RuntimeError(f"[E] encoded feature must be 2D [B,D], got shape {tuple(z.shape)}")

            zs.append(z.detach().cpu())
            coords_all.append(coords_np)

    if len(zs) == 0:
        raise RuntimeError(

        )

    Z = torch.cat(zs, dim=0)                       # [N, D]
    if Z.ndim != 2:
        Z = Z.view(Z.size(0), -1)
    coords = np.concatenate(coords_all, axis=0)    # [N, 2]

    if verbose > 0:
        print(f"[INFO] collected Z shape={tuple(Z.shape)}, coords shape={coords.shape}")
    Y = Z.detach().cpu().numpy().astype(np.float32)
    asset_dict = {'features': Y, 'coords': coords.astype(np.int32)}
    save_hdf5(output_path, asset_dict, attr_dict=None, mode='w')
    return output_path


def scan_slide_paths(dirs, recursive, exts):
    index = {}
    for d in dirs:
        root = Path(d)
        if not root.exists():
            print(f"[WARN] slide dir not found: {d}")
            continue
        paths = []
        if recursive:
            for ext in exts:
                paths += list(root.rglob(f"*{ext}"))
        else:
            for ext in exts:
                paths += list(root.glob(f"*{ext}"))
        for p in paths:
            if not p.is_file():
                continue
            slide_id = p.stem
            index[slide_id] = str(p.resolve())
    return index


def main(args):
    print('initializing dataset')
    if args.csv_path is None:
        raise NotImplementedError("csv_path must be provided")

    bags_dataset = Dataset_All_Bags(args.csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    pt_dir = os.path.join(args.feat_dir, 'pt_files')
    h5_dir = os.path.join(args.feat_dir, 'h5_files')
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(h5_dir, exist_ok=True)
    existing_pt = set(os.listdir(pt_dir))

    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)

    num_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.data_parallel and device.type == "cuda" and num_visible > 1:
        print(f"[INFO] Enabling DataParallel on {num_visible} GPUs (per-patch encoding only)")
        model = nn.DataParallel(model)
    else:
        if device.type == "cuda":
            print(f"[INFO] Using single GPU (visible={num_visible}), DataParallel disabled")
        else:
            print("[INFO] Using CPU")

    model.eval()
    model = model.to(device)

    loader_kwargs = {'num_workers': 8, 'pin_memory': (device.type == "cuda")}

    slide_dirs = [d.strip() for d in args.data_slide_dirs.split(",") if d.strip()]
    exts = tuple([e.strip() for e in args.exts.split(",") if e.strip()])

    print(f"[INFO] scanning slide directories (recursive={args.recursive}) ...")
    slide_index = scan_slide_paths(slide_dirs, recursive=args.recursive, exts=exts)
    if not slide_index:
        print("[ERROR] no slide files found from given data_slide_dirs")
        return

    total = len(bags_dataset)
    print(f"[INFO] total slides listed in csv: {total}")
    processed = 0

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    for bag_candidate_idx in tqdm(range(total)):
        raw_name = bags_dataset[bag_candidate_idx]
        slide_id = raw_name.split(args.slide_ext)[0] if raw_name.endswith(args.slide_ext) else Path(raw_name).stem

        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)

        if not os.path.isfile(h5_file_path):
            print(f"[WARN] coords h5 not found for slide {slide_id}: {h5_file_path}")
            continue

        slide_file_path = slide_index.get(slide_id, None)
        if slide_file_path is None or not os.path.isfile(slide_file_path):
            print(f"[WARN] slide file not found for {slide_id} in provided data_slide_dirs")
            continue

        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        pt_name = slide_id + '.pt'
        if not args.no_auto_skip and pt_name in existing_pt:
            print(f'skipped {slide_id}')
            continue

        output_path = os.path.join(h5_dir, bag_name)
        time_start = time.time()

        wsi = openslide.open_slide(slide_file_path)

        dataset = Whole_Slide_Bag_FP(
            file_path=h5_file_path,
            wsi=wsi,
            img_transforms=img_transforms
        )
        if len(dataset) == 0:
            print(f"[ERROR] no patches in {h5_file_path}")
            try:
                wsi.close()
            except Exception:
                pass
            continue

        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

        output_file_path = compute_slide_features_only(
            output_path=output_path,
            loader=loader,
            model=model,
            verbose=1
        )

        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {:.3f} s'.format(output_file_path, time_elapsed))

        with h5py.File(output_path, "r") as file:
            features = file['features'][:]
            print('features size: ', features.shape)
            print('coordinates size: ', file['coords'].shape)
        features = torch.from_numpy(features)
        torch.save(features, os.path.join(pt_dir, pt_name))
        existing_pt.add(pt_name)
        processed += 1

        try:
            wsi.close()
        except Exception:
            pass

    print(f"[INFO] done. processed {processed}/{total} slides.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extraction (per-patch VisionMamba only, no local attention)')
    parser.add_argument('--data_h5_dir', type=str, required=True, help="directory that contains 'patches/' with coords h5 files")
    parser.add_argument('--data_slide_dirs', type=str, required=True,
                        help="comma-separated directories containing slide files, e.g. /KICH,/KIRC,/KIRP")
    parser.add_argument('--recursive', action='store_true', help="recursively search slide files in data_slide_dirs")
    parser.add_argument('--exts', type=str, default='.svs,.SVS', help="comma-separated slide extensions")
    parser.add_argument('--slide_ext', type=str, default='.svs', help="expected extension when parsing csv slide names")
    parser.add_argument('--csv_path', type=str, required=True, help="csv containing slide list")
    parser.add_argument('--feat_dir', type=str, required=True, help="output features directory")
    parser.add_argument('--model_name', type=str, default='vmamba_ctx_v1',
                        choices=['vmamba_ctx_v1'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--target_patch_size', type=int, default=224)
    parser.add_argument('--data_parallel', action='store_true', help='enable nn.DataParallel if multiple GPUs')

    args = parser.parse_args()
    main(args)
