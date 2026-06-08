#!/usr/bin/env python3
"""실 TOW-IDS 데이터셋 → dataset_train.npz / dataset_test.npz (Team A / 김민세).

데이터 포맷 (tow-ids-dataset/):
  - *_training.pcap / *_test.pcap : per-packet 트래픽 (train/test 사전 분리)
  - y_train.csv / y_test.csv      : 헤더 없는 (idx, binary, multiclass) — 패킷과 1:1
    col3(multiclass) ∈ {Normal, F_I, P_I, M_F, C_D, C_R} = 우리 라벨명과 동일

파이프라인: 패킷 raw bytes → 64바이트 패딩/절단 → 연속 64패킷(stride 64, 비중첩)
            → 64x64 이미지 → /255 → 3-wavelet(coif1/db3/rbio1.3) LL → (3,32,32).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.imaging import frames_to_images  # noqa: E402
from src.preprocessing.pcap_parser import parse_pcap  # noqa: E402
from src.preprocessing.wavelet import wavelet_batch  # noqa: E402
from src.utils.io import LABEL_MAP, LABEL_NAMES, NUM_CLASSES, build_meta, save_dataset  # noqa: E402

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
    frames = parse_pcap(pcap, max_packets=max_packets)
    labels = _load_labels(csv, n_expect=len(frames), max_rows=max_packets)
    if not frames:
        raise ValueError(f'{pcap.name}: no packets were parsed')
    print(f"  packets={len(frames)}  imaging({cfg['packets']}x{cfg['bytes']}, "
          f"stride {cfg['packets']}) ...")

    imgs, ilabels, _ = frames_to_images(
        frames, labels,
        packets_per_image=cfg["packets"], bytes_per_packet=cfg["bytes"],
        stride=cfg["packets"], group_id=0)
    print(f"  windows={len(imgs)}  wavelet(3ch) ...")
    if len(imgs) == 0:
        raise ValueError(f'{pcap.name}: not enough packets for one image')
    X = wavelet_batch(imgs, names=("coif1", "db3", "rbio1.3"), level=cfg["level"])

    pcap_id = np.zeros(len(X), dtype=np.int64)
    packet_start = np.arange(len(X), dtype=np.int64) * int(cfg['packets'])
    packet_end = packet_start + int(cfg['packets'])

    meta = build_meta(X.shape[2], X.shape[3], seed=seed, git_sha=_git_sha(),
                      created_at=_dt.datetime.now().isoformat(timespec="seconds"),
                      stub=False, source=pcap.name,
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
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--packets", type=int, default=64)
    ap.add_argument("--bytes", type=int, default=64)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--max-packets", type=int, default=None, help="스모크용 패킷 제한")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = {"packets": args.packets, "bytes": args.bytes, "level": args.level}
    out = Path(args.out_dir)

    build_split(TRAIN_PCAP, DS / "y_train.csv",  out / "dataset_train.npz",
                cfg, args.max_packets, args.seed)
    build_split(TEST_PCAP, DS / "y_test.csv", out / "dataset_test.npz",
                cfg, args.max_packets, args.seed)


if __name__ == "__main__":
    main()
