"""PCAP 파싱 (Team A / 김민세) — dpkt 기반 고속 파서.

각 패킷에서 설정된 byte 영역(payload 또는 full Ethernet frame)을 반환한다.
"""
from __future__ import annotations

from pathlib import Path

import dpkt

ETH_HEADER_LEN = 14


def parse_pcap(
    path: str | Path,
    frame_content: str = 'payload',
    max_packets: int | None = None,
    strip_l2: bool | None = None,
) -> list[bytes]:
    """pcap/pcapng → 패킷별 raw 바이트 리스트."""
    if max_packets is not None and max_packets < 0:
        raise ValueError(f'max_packets must be non-negative, got {max_packets}')
    if strip_l2 is not None:
        frame_content = 'l3' if strip_l2 else 'full_frame'
    if frame_content not in {'payload', 'full_frame', 'l3'}:
        raise ValueError(
            f'frame_content must be payload, full_frame, or l3, got {frame_content}')
    path = Path(path)
    frames: list[bytes] = []
    try:
        with path.open('rb') as file:
            try:
                reader = dpkt.pcap.Reader(file)
            except ValueError:
                file.seek(0)
                reader = dpkt.pcapng.Reader(file)
            for index, (_timestamp, buffer) in enumerate(reader):
                if max_packets is not None and index >= max_packets:
                    break
                frame = bytes(buffer)
                if frame_content == 'full_frame':
                    frames.append(frame)
                    continue
                try:
                    ethernet = dpkt.ethernet.Ethernet(frame)
                    layer = ethernet.data
                    if frame_content == 'l3':
                        frames.append(bytes(layer))
                        continue
                    while hasattr(layer, 'data') and not isinstance(layer, bytes):
                        next_layer = layer.data
                        if next_layer is layer:
                            break
                        layer = next_layer
                    frames.append(bytes(layer))
                except (ValueError, dpkt.dpkt.Error):
                    frames.append(frame[ETH_HEADER_LEN:])
    except (OSError, ValueError, dpkt.dpkt.Error) as exc:
        raise ValueError(f'failed to parse capture {path}: {exc}') from exc
    return frames


def write_pcap(path: str | Path, frames: list[bytes]) -> None:
    """테스트/스모크용 합성 pcap 작성."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as file:
        writer = dpkt.pcap.Writer(file)
        for i, fr in enumerate(frames):
            writer.writepkt(fr, ts=float(i))
