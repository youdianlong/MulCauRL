# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df

# other imports
import os
import sys
import numpy as np
import time
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
    total_time = time.time() - start
    return heatmap, total_time

def segment(WSI_object, seg_params=None, filter_params=None, mask_file=None):
    start_time = time.time()
    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
    start_time = time.time()
    file_path = WSI_object.process_contours(**kwargs)
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed

def collect_slides(sources, recursive=False, exts=(".svs", ".SVS")):
    """
    Collect all slide paths from multiple source directories.
    Returns:
        slides: list of slide file names (basename, with auto suffix if duplicated)
        slide_to_path: dict mapping basename -> full path
    """
    slides = []
    slide_to_path = {}
    seen = {}

    for src in sources:
        src_path = Path(src)
        if not src_path.exists():
            print(f"[WARN] source does not exist: {src}")
            continue

        if recursive:
            candidates = []
            for ext in exts:
                candidates.extend(src_path.rglob(f"*{ext}"))
        else:
            candidates = []
            for ext in exts:
                candidates.extend(src_path.glob(f"*{ext}"))

        for p in candidates:
            if not p.is_file():
                continue
            base = p.name  # e.g., XYZ.svs

            # handle duplicate basenames across sources
            if base in seen:
                seen[base] += 1
                stem, suf = base.rsplit(".", 1)
                base_uniq = f"{stem}__{seen[base]:03d}.{suf}"
            else:
                seen[base] = 0
                base_uniq = base

            slides.append(base_uniq)
            slide_to_path[base_uniq] = str(p)

    return slides, slide_to_path

def safe_unique_path(dest_dir: Path, name: str) -> Path:
    """
    Return a non-existing path within dest_dir using name, appending a numeric suffix if needed.
    """
    p = dest_dir / name
    if not p.exists():
        return p
    stem = p.stem
    suffix = p.suffix
    i = 1
    while True:
        q = dest_dir / f"{stem}__{i:03d}{suffix}"
        if not q.exists():
            return q
        i += 1

def seg_and_patch(
    save_dir,
    patch_save_dir,
    mask_save_dir,
    stitch_save_dir,
    patch_size=256,
    step_size=256,
    seg_params=None,
    filter_params=None,
    vis_params=None,
    patch_params=None,
    patch_level=0,
    use_default_params=False,
    seg=False,
    save_mask=True,
    stitch=False,
    patch=False,
    auto_skip=True,
    # new inputs
    slides=None,
    slide_to_path=None,
    # restored compatibility
    process_list=None
):
    """
    slides: list of slide basenames (may include auto-suffixed names)
    slide_to_path: dict {basename: full_path}
    process_list: optional CSV path. If provided, df is initialized from it (as original behavior).
                  Its 'slide_id' column values must match names we can map to real files:
                  - if they match collected names directly, use them;
                  - else if they match raw basenames, we try to resolve by searching in slide_to_path keys.
    """
    # defaults
    if seg_params is None:
        seg_params = {"seg_level": -1, "sthresh": 8, "mthresh": 7, "close": 4, "use_otsu": False, "keep_ids": "none", "exclude_ids": "none"}
    if filter_params is None:
        filter_params = {"a_t": 100, "a_h": 16, "max_n_holes": 8}
    if vis_params is None:
        vis_params = {"vis_level": -1, "line_thickness": 500}
    if patch_params is None:
        patch_params = {"use_padding": True, "contour_fn": "four_pt"}

    if process_list is not None:
        if not os.path.isfile(process_list):
            print(f"[ERROR] process_list not found: {process_list}")
            return 0.0, 0.0
        df_in = pd.read_csv(process_list)
        df = initialize_df(df_in, seg_params, filter_params, vis_params, patch_params)
        # adapt mapping: ensure each df slide_id is mapped to a full path
        if slides is None or slide_to_path is None:
            print("[WARN] process_list provided without slides/slide_to_path; attempting to infer later may fail.")
    else:
        if slides is None or slide_to_path is None:
            print("[ERROR] slides/slide_to_path must be provided when process_list is not used.")
            return 0.0, 0.0
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)

    mask = df["process"] == 1
    process_stack = df[mask]
    total = len(process_stack)

    if total == 0:
        print("[INFO] No slides to process. Please check your sources/filters.")
        return 0.0, 0.0

    legacy_support = "a" in df.keys()
    if legacy_support:
        print("detected legacy segmentation csv file, legacy support enabled")
        df = df.assign(
            **{
                "a_t": np.full((len(df)), int(filter_params["a_t"]), dtype=np.uint32),
                "a_h": np.full((len(df)), int(filter_params["a_h"]), dtype=np.uint32),
                "max_n_holes": np.full((len(df)), int(filter_params["max_n_holes"]), dtype=np.uint32),
                "line_thickness": np.full((len(df)), int(vis_params["line_thickness"]), dtype=np.uint32),
                "contour_fn": np.full((len(df)), patch_params["contour_fn"]),
            }
        )

    seg_times = 0.0
    patch_times = 0.0
    stitch_times = 0.0

    for i in tqdm(range(total)):
        df.to_csv(os.path.join(save_dir, "process_list_autogen.csv"), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, "slide_id"]

        print("\n\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
        print("processing {}".format(slide))

        df.loc[idx, "process"] = 0
        slide_id, _ = os.path.splitext(slide)

        # resolve full path
        full_path = None
        if slide_to_path and slide in slide_to_path:
            full_path = slide_to_path[slide]
        elif slide_to_path:
            # try to match raw basename (without any "__NNN" suffix that might be in collected list)
            # find any key whose stem equals slide_id
            keys = list(slide_to_path.keys())
            candidates = [k for k in keys if os.path.splitext(k)[0] == slide_id]
            if len(candidates) == 1:
                full_path = slide_to_path[candidates[0]]

        if full_path is None or not os.path.isfile(full_path):
            print(f"[WARN] slide path not found or not a file: {slide}")
            df.loc[idx, "status"] = "missing_path"
            continue

        # auto-skip if exists
        dest_h5 = Path(patch_save_dir) / f"{slide_id}.h5"
        if auto_skip and dest_h5.exists():
            print(f"{slide_id} already exist in destination location, skipped")
            df.loc[idx, "status"] = "already_exist"
            continue

        # Initialize WSI
        WSI_object = WholeSlideImage(full_path)

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()
        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}

            for key in vis_params.keys():
                if legacy_support and key == "vis_level":
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == "a_t":
                    old_area = df.loc[idx, "a"]
                    seg_level = df.loc[idx, "seg_level"]
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == "seg_level":
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        # resolve vis/seg levels if negative
        if current_vis_params["vis_level"] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params["vis_level"] = 0
            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params["vis_level"] = best_level

        if current_seg_params["seg_level"] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params["seg_level"] = 0
            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params["seg_level"] = best_level

        keep_ids = str(current_seg_params["keep_ids"])
        if keep_ids != "none" and len(keep_ids) > 0:
            str_ids = current_seg_params["keep_ids"]
            current_seg_params["keep_ids"] = np.array(str_ids.split(",")).astype(int)
        else:
            current_seg_params["keep_ids"] = []

        exclude_ids = str(current_seg_params["exclude_ids"])
        if exclude_ids != "none" and len(exclude_ids) > 0:
            str_ids = current_seg_params["exclude_ids"]
            current_seg_params["exclude_ids"] = np.array(str_ids.split(",")).astype(int)
        else:
            current_seg_params["exclude_ids"] = []

        w, h = WSI_object.level_dim[current_seg_params["seg_level"]]
        if w * h > 1e8:
            print("level_dim {} x {} is likely too large for successful segmentation, aborting".format(w, h))
            df.loc[idx, "status"] = "failed_seg"
            continue

        df.loc[idx, "vis_level"] = current_vis_params["vis_level"]
        df.loc[idx, "seg_level"] = current_seg_params["seg_level"]

        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)

        if save_mask:
            mask_img = WSI_object.visWSI(**current_vis_params)
            mask_path = Path(mask_save_dir) / f"{slide_id}.jpg"
            mask_path = safe_unique_path(Path(mask_save_dir), mask_path.name)
            mask_img.save(str(mask_path))

        patch_time_elapsed = -1
        if patch:
            current_patch_params.update(
                {"patch_level": patch_level, "patch_size": patch_size, "step_size": step_size, "save_path": patch_save_dir}
            )
            _, patch_time_elapsed = patching(WSI_object=WSI_object, **current_patch_params)

        stitch_time_elapsed = -1
        if stitch:
            file_path = Path(patch_save_dir) / f"{slide_id}.h5"
            if file_path.is_file():
                heatmap, stitch_time_elapsed = stitching(str(file_path), WSI_object, downscale=64)
                stitch_path = safe_unique_path(Path(stitch_save_dir), f"{slide_id}.jpg")
                heatmap.save(str(stitch_path))

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))
        df.loc[idx, "status"] = "processed"

        seg_times += 0 if seg_time_elapsed < 0 else seg_time_elapsed
        patch_times += 0 if patch_time_elapsed < 0 else patch_time_elapsed
        stitch_times += 0 if stitch_time_elapsed < 0 else stitch_time_elapsed

    if total > 0:
        seg_times /= total
        patch_times /= total
        stitch_times = stitch_times / total if total > 0 else 0.0

    df.to_csv(os.path.join(save_dir, "process_list_autogen.csv"), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))
    return seg_times, patch_times

def parse_args():
    parser = argparse.ArgumentParser(description="seg and patch (multi-source)")
    # single source (backward compatibility)
    parser.add_argument("--source", type=str, default=None, help="single source directory")
    # multiple sources
    parser.add_argument("--sources", type=str, default=None, help="comma-separated directories, e.g. /KICH,/KIRC,/KIRP")
    parser.add_argument("--recursive", action="store_true", help="recursively search in sources")
    parser.add_argument("--exts", type=str, default=".svs,.SVS", help="comma-separated extensions (default: .svs,.SVS)")

    parser.add_argument("--step_size", type=int, default=256, help="step_size")
    parser.add_argument("--patch_size", type=int, default=256, help="patch_size")
    parser.add_argument("--patch", default=False, action="store_true")
    parser.add_argument("--seg", default=False, action="store_true")
    parser.add_argument("--stitch", default=False, action="store_true")
    parser.add_argument("--no_auto_skip", default=True, action="store_false")
    parser.add_argument("--save_dir", type=str, required=True, help="directory to save processed data")
    parser.add_argument("--preset", default=None, type=str, help="predefined parameters profile (.csv)")
    parser.add_argument("--patch_level", type=int, default=0, help="downsample level at which to patch")
    parser.add_argument("--process_list", type=str, default=None, help="csv of images to process with parameters")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # resolve sources
    if args.sources:
        source_dirs = [s.strip() for s in args.sources.split(",") if s.strip()]
    else:
        if not args.source:
            print("[ERROR] you must specify --sources or --source")
            sys.exit(1)
        source_dirs = [args.source]

    patch_save_dir = os.path.join(args.save_dir, "patches")
    mask_save_dir = os.path.join(args.save_dir, "masks")
    stitch_save_dir = os.path.join(args.save_dir, "stitches")

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list) if not os.path.isabs(args.process_list) else args.process_list
    else:
        process_list = None

    print("sources: ", source_dirs)
    print("patch_save_dir: ", patch_save_dir)
    print("mask_save_dir: ", mask_save_dir)
    print("stitch_save_dir: ", stitch_save_dir)

    directories = {
        "save_dir": args.save_dir,
        "patch_save_dir": patch_save_dir,
        "mask_save_dir": mask_save_dir,
        "stitch_save_dir": stitch_save_dir,
    }

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        os.makedirs(val, exist_ok=True)

    seg_params = {"seg_level": -1, "sthresh": 8, "mthresh": 7, "close": 4, "use_otsu": False, "keep_ids": "none", "exclude_ids": "none"}
    filter_params = {"a_t": 100, "a_h": 16, "max_n_holes": 8}
    vis_params = {"vis_level": -1, "line_thickness": 250}
    patch_params = {"use_padding": True, "contour_fn": "four_pt"}

    if args.preset:
        preset_df = pd.read_csv(os.path.join("presets", args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]
        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]
        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]
        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {"seg_params": seg_params, "filter_params": filter_params, "patch_params": patch_params, "vis_params": vis_params}
    print(parameters)

    # collect slides
    exts = tuple([e.strip() for e in args.exts.split(",") if e.strip()])
    slides, slide_to_path = collect_slides(source_dirs, recursive=args.recursive, exts=exts)

    if len(slides) == 0 and not args.process_list:
        print("[INFO] No slides found in the provided sources. Nothing to do.")
        sys.exit(0)

    # run
    seg_times, patch_times = seg_and_patch(
        **directories,
        **parameters,
        patch_size=args.patch_size,
        step_size=args.step_size,
        seg=args.seg,
        use_default_params=False,
        save_mask=True,
        stitch=args.stitch,
        patch_level=args.patch_level,
        patch=args.patch,
        process_list=process_list,
        auto_skip=args.no_auto_skip,
        slides=slides if len(slides) > 0 else None,
        slide_to_path=slide_to_path if len(slides) > 0 else None,
    )
