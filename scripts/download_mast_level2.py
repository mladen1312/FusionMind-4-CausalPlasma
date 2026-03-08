#!/usr/bin/env python3
"""
FAIR-MAST Level 2 Bulk Downloader — run locally to download 3000+ shots.

Usage:
    python scripts/download_mast_level2.py [--target 3000] [--batch-size 20]

Resumes automatically. Saves batches to data/mast/batches/, merges into
data/mast/mast_level2_all.npz when done.

Estimated: ~7s/shot → 3000 shots ≈ 6 hours. Can be interrupted and resumed.
"""
import s3fs, zarr, numpy as np, json, time, warnings, glob, os, argparse
from scipy.interpolate import interp1d
from pathlib import Path
warnings.filterwarnings("ignore")

L2V = {
    'betan': 'equilibrium/beta_tor_normal', 'betap': 'equilibrium/beta_pol',
    'betat': 'equilibrium/beta_tor', 'q95': 'equilibrium/q95',
    'elongation': 'equilibrium/elongation', 'li': 'equilibrium/li',
    'wmhd': 'equilibrium/wmhd', 'q_axis': 'equilibrium/q_axis',
    'minor_radius': 'equilibrium/minor_radius',
    'tribot': 'equilibrium/triangularity_lower', 'tritop': 'equilibrium/triangularity_upper',
    'Ip': 'summary/ip', 'ne_line': 'summary/line_average_n_e',
    'greenwald_den': 'summary/greenwald_density',
    'p_rad': 'summary/power_radiated', 'p_nbi': 'summary/power_nbi'
}
VN = list(L2V.keys())

BATCH_DIR = Path("data/mast/batches")
S3_LIST = Path("data/mast/mast_l2_all_shots_on_s3.json")
MERGED = Path("data/mast/mast_level2_all.npz")
LABELS = Path("data/mast/mast_labels_all.json")


def get_s3_shot_list(fs):
    """Get or cache list of all Level 2 shots on S3."""
    if S3_LIST.exists():
        with open(S3_LIST) as f:
            return json.load(f)
    print("Scanning S3 for available shots...")
    entries = fs.ls('mast/level2/shots/', detail=False)
    shots = sorted(int(e.split('/')[-1].replace('.zarr', ''))
                   for e in entries if e.endswith('.zarr'))
    S3_LIST.parent.mkdir(parents=True, exist_ok=True)
    with open(S3_LIST, 'w') as f:
        json.dump(shots, f)
    print(f"Found {len(shots)} shots on S3")
    return shots


def get_done_shots():
    """Get set of already-downloaded shot IDs."""
    done = set()
    for f in sorted(glob.glob(str(BATCH_DIR / "batch_*.npz"))):
        d = np.load(f, allow_pickle=True)
        done |= set(np.unique(d['shot_ids']).astype(int))
    return done


def download_shot(fs, sid):
    """Download one shot, return (data_array, label) or None."""
    try:
        store = s3fs.S3Map(root=f'mast/level2/shots/{sid}.zarr', s3=fs, check=False)
        root = zarr.open(store, mode='r')
        t_eq = np.array(root['equilibrium']['time'])
        m = t_eq > 0.01; t_eq = t_eq[m]
        if len(t_eq) < 10:
            return None
        t_sum = np.array(root['summary']['time']) if 'time' in root['summary'] else t_eq

        cols = []
        for vn, path in L2V.items():
            grp, sig = path.split('/')[0], path.split('/')[-1]
            try:
                data = np.array(root[grp][sig])
                t_src = t_eq if grp == 'equilibrium' else t_sum
                if grp == 'equilibrium':
                    col = data[m]
                else:
                    v = np.isfinite(data) & np.isfinite(t_src)
                    if v.sum() > 5:
                        col = np.nan_to_num(interp1d(t_src[v], data[v],
                                                      bounds_error=False, fill_value=0)(t_eq))
                    else:
                        col = np.zeros(len(t_eq))
            except:
                col = np.zeros(len(t_eq))
            cols.append(col)

        ext = np.nan_to_num(np.column_stack(cols)).astype(np.float32)
        ext = ext[np.all(np.isfinite(ext), axis=1)]
        if len(ext) < 10:
            return None
        return ext
    except:
        return None


def merge_all():
    """Merge all batches into single file."""
    D_parts, L_parts = [], []
    labels = {'disrupted': [], 'clean': []}

    for f in sorted(glob.glob(str(BATCH_DIR / "batch_*.npz"))):
        d = np.load(f, allow_pickle=True)
        D_parts.append(d['data']); L_parts.append(d['shot_ids'])

    lbl_files = sorted(glob.glob(str(BATCH_DIR / "labels_*.json")))
    for f in lbl_files:
        with open(f) as fh:
            batch_labels = json.load(fh)
            for entry in batch_labels:
                labels.setdefault(entry['label'], []).append(entry['sid'])

    if not D_parts:
        print("No data to merge")
        return

    D = np.vstack(D_parts); L = np.concatenate(L_parts)
    np.savez_compressed(MERGED, data=D, shot_ids=L, variables=np.array(VN))
    with open(LABELS, 'w') as f:
        json.dump(labels, f)
    print(f"Merged: {len(np.unique(L))} shots, {D.shape[0]} timepoints → {MERGED}")


def main():
    parser = argparse.ArgumentParser(description="Download FAIR-MAST Level 2 data")
    parser.add_argument('--target', type=int, default=3000, help='Target number of shots')
    parser.add_argument('--batch-size', type=int, default=20, help='Shots per batch file')
    parser.add_argument('--merge-only', action='store_true', help='Just merge existing batches')
    args = parser.parse_args()

    if args.merge_only:
        merge_all()
        return

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    fs = s3fs.S3FileSystem(anon=True, client_kwargs={
        'endpoint_url': 'https://s3.echo.stfc.ac.uk', 'region_name': 'us-east-1'},
        config_kwargs={'connect_timeout': 10, 'read_timeout': 15})

    all_s3 = get_s3_shot_list(fs)
    done = get_done_shots()
    avail = sorted([s for s in all_s3 if s not in done and s >= 24000])

    print(f"Target: {args.target} | Downloaded: {len(done)} | Available: {len(avail)}")
    remaining = args.target - len(done)
    if remaining <= 0:
        print("Target reached! Merging...")
        merge_all()
        return

    print(f"Downloading {min(remaining, len(avail))} more shots...")
    t_total = time.time()
    batch_num = len(glob.glob(str(BATCH_DIR / "batch_*.npz")))

    all_d, all_l, nl = [], [], []
    for i, sid in enumerate(avail[:remaining]):
        data = download_shot(fs, sid)
        if data is not None:
            all_d.append(data)
            all_l.append(np.full(len(data), sid, dtype=np.int32))
            nl.append({'sid': int(sid), 'label': 'clean'})

        # Save batch
        if len(all_d) >= args.batch_size or (i == min(remaining, len(avail)) - 1 and all_d):
            batch_num += 1
            np.savez_compressed(BATCH_DIR / f"batch_{batch_num:04d}.npz",
                                data=np.vstack(all_d), shot_ids=np.concatenate(all_l),
                                variables=np.array(VN))
            with open(BATCH_DIR / f"labels_{batch_num:04d}.json", 'w') as f:
                json.dump(nl, f)

            elapsed = time.time() - t_total
            total_done = len(done) + sum(len(d) for d in all_d)
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (remaining - i - 1) / rate / 3600 if rate > 0 else 0
            print(f"  Batch {batch_num}: +{len(all_d)} shots "
                  f"({len(done) + i + 1}/{args.target}) "
                  f"[{rate:.1f} shots/s, ETA {eta:.1f}h]")
            all_d, all_l, nl = [], [], []

    print(f"\nDone! Total time: {(time.time()-t_total)/3600:.1f}h")
    merge_all()


if __name__ == "__main__":
    main()
