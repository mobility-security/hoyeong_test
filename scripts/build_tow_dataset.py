#!/usr/bin/env python3
"""실 TOW-IDS 데이터셋 → dataset_train.npz / dataset_test.npz (Team A / 김민세).

데이터 포맷 (tow-ids-dataset/):
  - *_training.pcap / *_test.pcap : per-packet 트래픽 (train/test 사전 분리)
  - y_train.csv / y_test.csv      : 헤더 없는 (idx, binary, multiclass) — 패킷과 1:1
    col3(multiclass) ∈ {Normal, F_I, P_I, M_F, C_D, C_R} = 우리 라벨명과 동일

파이프라인: 패킷 raw bytes → 128바이트 패딩/절단 → 연속 128패킷(stride 128, 비중첩)
            → 128x128 이미지 → /255 → 3-wavelet(coif1/db3/rbio1.3) LL → (3,64,64).
윈도우 라벨: 윈도우 내 공격이 있으면 최빈 공격, 없으면 Normal (보안 보수적).
pcap_id   : 실제 원본 capture id. 이 스크립트는 split마다 PCAP 하나를 읽으므로 모두 0.
packet_start/end: 각 이미지가 사용한 원본 패킷의 [start, end) 범위.

사용:
    python scripts/build_tow_dataset.py            # 전체 빌드
    python scripts/build_tow_dataset.py --max-packets 200000   # 빠른 스모크
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.imaging import frames_to_images, window_packet_ranges  # noqa: E402
from src.preprocessing.pcap_parser import parse_pcap  # noqa: E402
from src.preprocessing.wavelet import wavelet_batch  # noqa: E402
from src.utils.io import (  # noqa: E402
    LABEL_MAP, LABEL_NAMES, NUM_CLASSES, build_meta, save_dataset, sha256_file,
)

DS = Path("data/raw")
TRAIN_PCAP = DS / "Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap"
TEST_PCAP = DS / "Automotive_Ethernet_with_Attack_original_10_17_20_04_test.pcap"


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_labels(
    csv: Path,
    n_expect: int | None = None,
    max_rows: int | None = None,
) -> np.ndarray:
    """y_*.csv 의 multiclass(col3) → int64 라벨 배열."""
    df = pd.read_csv(csv, header=None, usecols=[2], names=['mc'], nrows=max_rows)
    unknown = set(df["mc"].unique()) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"{csv.name}: 미지의 라벨 {unknown} (LABEL_MAP 와 불일치)")
    y = df["mc"].map(LABEL_MAP).to_numpy(dtype=np.int64)
    if n_expect is not None and len(y) != n_expect:
        raise ValueError(f"{csv.name}: 라벨 {len(y)} != 패킷 {n_expect} (정렬 깨짐)")
    return y


def build_split(pcap: Path, csv: Path, out: Path, cfg,
                max_packets: int | None, seed: int):
    print(f"[{out.name}] parsing {pcap.name} ...")
    frame_content = str(cfg['frame_content'])
    pad_partial = str(cfg['partial_window']) == 'pad'
    frames = parse_pcap(
        pcap, frame_content=frame_content, max_packets=max_packets)
    labels = _load_labels(csv, n_expect=len(frames), max_rows=max_packets)
    if not frames:
        raise ValueError(f'{pcap.name}: no packets were parsed')
    print(f"  packets={len(frames)}  imaging({cfg['packets']}x{cfg['bytes']}, "
          f"stride {cfg['packets']}) ...")

    imgs, ilabels, _ = frames_to_images(
        frames, labels,
        packets_per_image=cfg["packets"], bytes_per_packet=cfg["bytes"],
        stride=cfg["stride"], group_id=0, pad_partial=pad_partial)
    print(f"  windows={len(imgs)}  wavelet(3ch) ...")
    if len(imgs) == 0:
        raise ValueError(f'{pcap.name}: not enough packets for one image')
    X = wavelet_batch(
        imgs, names=tuple(cfg['wavelets']), level=cfg["level"],
        mode=cfg['wavelet_mode'], channel_norm=cfg['channel_norm'])

    pcap_id = np.zeros(len(X), dtype=np.int64)
    packet_start, packet_end = window_packet_ranges(
        len(frames), int(cfg['packets']), int(cfg['stride']), pad_partial)

    meta = build_meta(
                      X.shape[2], X.shape[3], wavelets=tuple(cfg['wavelets']),
                      mode=cfg['wavelet_mode'], norm=cfg['channel_norm'],
                      seed=seed, git_sha=_git_sha(),
                      created_at=_dt.datetime.now().isoformat(timespec="seconds"),
                      packets_per_image=cfg['packets'],
                      bytes_per_packet=cfg['bytes'], stride=cfg['stride'],
                      frame_content=frame_content,
                      partial_window=str(cfg['partial_window']),
                      stub=False, source=pcap.name,
                      source_pcap_sha256=sha256_file(pcap),
                      source_labels_sha256=sha256_file(csv),
                      group_semantics='capture',
                      packet_range_semantics='zero_based_half_open',
                      note="real TOW-IDS dataset; window label = dominant attack if any")
    o = save_dataset(
        out, X, ilabels, meta, pcap_id=pcap_id,
        packet_start=packet_start, packet_end=packet_end)
    counts = {LABEL_NAMES[c]: int((ilabels == c).sum()) for c in range(NUM_CLASSES)}
    print(f"  saved: {o}  (+ {o.with_suffix('.meta.json').name})")
    print(f"  X={X.shape}  captures={len(np.unique(pcap_id))}  window-counts={counts}\n")
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--packets", type=int, default=None)
    ap.add_argument("--bytes", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--max-packets", type=int, default=None, help="스모크용 패킷 제한")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    raw_cfg = OmegaConf.load(args.config)
    cfg = {
        "packets": args.packets or int(raw_cfg.imaging.packets_per_image),
        "bytes": args.bytes or int(raw_cfg.imaging.bytes_per_packet),
        "stride": args.stride or int(raw_cfg.imaging.stride),
        "level": args.level or int(raw_cfg.wavelet.level),
        "wavelets": list(raw_cfg.wavelet.names),
        "wavelet_mode": str(raw_cfg.wavelet.mode),
        "channel_norm": str(raw_cfg.wavelet.channel_norm),
        "frame_content": str(raw_cfg.imaging.frame_content),
        "partial_window": str(raw_cfg.imaging.partial_window),
    }
    out = Path(args.out_dir)

    build_split(TRAIN_PCAP, DS / "y_train.csv",  out / "dataset_train.npz",
                cfg, args.max_packets, args.seed)
    build_split(TEST_PCAP, DS / "y_test.csv", out / "dataset_test.npz",
                cfg, args.max_packets, args.seed)


if __name__ == "__main__":
    main()
