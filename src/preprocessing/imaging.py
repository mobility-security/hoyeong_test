"""이미징 (Team A / 김민세) — 패킷 바이트 → N×M 이미지 → /255 정규화.

연속 `packets_per_image` 개 패킷을 각 `bytes_per_packet` 바이트로 패딩/절단해
(packets_per_image × bytes_per_packet) 2D 이미지를 만든다. wavelet 이전 단계.
"""
from __future__ import annotations

import numpy as np

from src.utils.io import NUM_CLASSES


def payload_to_row(frame: bytes, n_bytes: int) -> np.ndarray:
    """프레임 바이트 → 길이 n_bytes 의 uint8 행 (zero-pad / truncate)."""
    if n_bytes <= 0:
        raise ValueError(f'n_bytes must be positive, got {n_bytes}')
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise TypeError(f'frame must be bytes-like, got {type(frame)}')
    arr = np.frombuffer(frame[:n_bytes], dtype=np.uint8)
    if arr.size < n_bytes:
        arr = np.pad(arr, (0, n_bytes - arr.size), constant_values=0)
    return arr


def _window_label(labels: np.ndarray) -> int:
    """윈도우 라벨 = 비정상이 하나라도 있으면 최빈 비정상, 아니면 Normal(0)."""
    nz = labels[labels != 0]
    if nz.size == 0:
        return 0
    return int(np.bincount(nz).argmax())


def frames_to_images(frames: list[bytes], labels, *, packets_per_image: int,
                     bytes_per_packet: int, stride: int | None = None,
                     group_id: int | None = None,
                     pad_partial: bool = True):
    """슬라이딩 윈도우 이미징.

    반환: images (M, P, B) float32 in [0,1], img_labels (M,) int64, img_groups (M,) int64.
    group_id 가 주어지면 모든 윈도우에 동일 group(=capture) 부여 → 누수 안전 split 용.
    """
    if packets_per_image <= 0:
        raise ValueError(f'packets_per_image must be positive, got {packets_per_image}')
    if bytes_per_packet <= 0:
        raise ValueError(f'bytes_per_packet must be positive, got {bytes_per_packet}')
    if stride is None:
        stride = packets_per_image
    if stride <= 0:
        raise ValueError(f'stride must be positive, got {stride}')
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f'labels must be 1-D, got {labels.shape}')
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f'labels must contain integers, got {labels.dtype}')
    labels = labels.astype(np.int64, copy=False)
    if len(frames) != len(labels):
        raise ValueError(f"len(frames)={len(frames)} != len(labels)={len(labels)}")
    if labels.size and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
        raise ValueError(f'labels must be in [0, {NUM_CLASSES - 1}]')
    if not frames:
        return (np.empty((0, packets_per_image, bytes_per_packet), np.float32),
                np.empty((0,), np.int64), np.empty((0,), np.int64))
    if len(frames) < packets_per_image and not pad_partial:
        return (np.empty((0, packets_per_image, bytes_per_packet), np.float32),
                np.empty((0,), np.int64), np.empty((0,), np.int64))

    rows = np.stack([payload_to_row(f, bytes_per_packet) for f in frames])  # (P,B) uint8
    imgs, img_labels, img_groups = [], [], []
    stop = len(rows) if pad_partial else len(rows) - packets_per_image + 1
    for s in range(0, stop, stride):
        actual_rows = rows[s:s + packets_per_image]
        if len(actual_rows) < packets_per_image:
            actual_rows = np.pad(
                actual_rows,
                ((0, packets_per_image - len(actual_rows)), (0, 0)),
                constant_values=0,
            )
        block = actual_rows.astype(np.float32) / 255.0  # [0,1]
        imgs.append(block)
        img_labels.append(_window_label(labels[s:min(s + packets_per_image, len(labels))]))
        img_groups.append(group_id if group_id is not None else s)
    return (np.asarray(imgs, dtype=np.float32),
            np.asarray(img_labels, dtype=np.int64),
            np.asarray(img_groups, dtype=np.int64))


def window_packet_ranges(
    n_packets: int,
    packets_per_image: int,
    stride: int,
    pad_partial: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact [start, end) packet provenance for generated windows."""
    if n_packets < 0 or packets_per_image <= 0 or stride <= 0:
        raise ValueError('window range arguments must be non-negative/positive')
    if n_packets == 0 or (n_packets < packets_per_image and not pad_partial):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    stop = n_packets if pad_partial else n_packets - packets_per_image + 1
    starts = np.arange(0, stop, stride, dtype=np.int64)
    ends = np.minimum(starts + packets_per_image, n_packets).astype(np.int64)
    return starts, ends
