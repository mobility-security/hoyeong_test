"""전처리 오케스트레이터 (Team A / 김민세) — PCAP 매니페스트 → dataset.npz v1.

매니페스트 CSV 컬럼: pcap_path, label_name   (label_name ∈ io.LABEL_MAP 키)
캡처는 시나리오별이라 한 pcap 의 모든 패킷에 그 pcap 의 라벨을 부여한다.
group_id = pcap 인덱스 → 누수 안전 split 의 atomic 단위.

사용:
    python -m src.preprocessing.build_dataset \
        --manifest data/raw/manifest.csv --out data/processed/dataset.npz
"""
from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.imaging import frames_to_images  # noqa: E402
from src.preprocessing.pcap_parser import parse_pcap  # noqa: E402
from src.preprocessing.wavelet import wavelet_batch  # noqa: E402
from src.utils.io import LABEL_MAP, LABEL_NAMES, NUM_CLASSES, build_meta, save_dataset  # noqa: E402


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(manifest_csv: str, cfg, out_path: str, seed: int = 0):
    df = pd.read_csv(manifest_csv)
    required_columns = {'pcap_path', 'label_name'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f'manifest missing columns: {sorted(missing_columns)}')
    if df.empty:
        raise ValueError('manifest contains no captures')
    if df[['pcap_path', 'label_name']].isna().any().any():
        raise ValueError('manifest pcap_path/label_name columns must not contain null values')
    unknown_labels = sorted(set(df['label_name'].astype(str)) - set(LABEL_MAP))
    if unknown_labels:
        raise ValueError(f'manifest contains unknown labels: {unknown_labels}')
    img_cfg, wav_cfg = cfg.imaging, cfg.wavelet
    Xs, ys, gs, starts, ends = [], [], [], [], []
    for gid, row in enumerate(df.itertuples(index=False)):
        label = LABEL_MAP[row.label_name]
        frames = parse_pcap(row.pcap_path)
        labels = np.full(len(frames), label, dtype=np.int64)
        imgs, ilabels, igroups = frames_to_images(
            frames, labels,
            packets_per_image=img_cfg.packets_per_image,
            bytes_per_packet=img_cfg.bytes_per_packet,
            stride=img_cfg.stride, group_id=gid)
        if imgs.shape[0] == 0:
            print(f"  [warn] {row.pcap_path}: 패킷 부족 → skip")
            continue
        Xs.append(wavelet_batch(
            imgs,
            names=tuple(wav_cfg.names),
            level=wav_cfg.level,
            mode=wav_cfg.mode,
            channel_norm=wav_cfg.channel_norm,
        ))
        ys.append(ilabels)
        gs.append(igroups)
        packet_start = np.arange(len(imgs), dtype=np.int64) * int(img_cfg.stride)
        starts.append(packet_start)
        ends.append(packet_start + int(img_cfg.packets_per_image))
        print(f"  {row.label_name:6s} {Path(row.pcap_path).name}: "
              f"{len(frames)} pkts → {imgs.shape[0]} imgs")

    if not Xs:
        raise ValueError('no capture produced enough packets for one image')

    X = np.concatenate(Xs, 0).astype(np.float32)
    y = np.concatenate(ys, 0).astype(np.int64)
    groups = np.concatenate(gs, 0).astype(np.int64)
    packet_start = np.concatenate(starts, 0).astype(np.int64)
    packet_end = np.concatenate(ends, 0).astype(np.int64)

    meta = build_meta(X.shape[2], X.shape[3], wavelets=list(wav_cfg.names),
                      mode=wav_cfg.mode, seed=seed, git_sha=_git_sha(),
                      created_at=_dt.datetime.now().isoformat(timespec="seconds"),
                      stub=False, group_semantics='capture',
                      packet_range_semantics='zero_based_half_open')
    out = save_dataset(
        out_path, X, y, meta, pcap_id=groups,
        packet_start=packet_start, packet_end=packet_end)
    counts = {LABEL_NAMES[c]: int((y == c).sum()) for c in range(NUM_CLASSES)}
    print(f"saved: {out}  (+ {out.with_suffix('.meta.json').name})")
    print(f"  X={X.shape}  pcap_id={len(np.unique(groups))}  counts={counts}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="pcap_path,label_name CSV")
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--out", default="data/processed/dataset.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    build(args.manifest, cfg, args.out, args.seed)


if __name__ == "__main__":
    main()
