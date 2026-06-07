"""PCAP 파싱 (Team A / 김민세) — dpkt 기반 고속 파서.

각 패킷의 raw Ethernet 프레임 바이트(헤더+payload)를 0-255 정수 시퀀스로 반환한다.
원본 TOW-IDS 는 payload 중심이나, byte-image IDS 에서 전체 프레임 바이트 사용은
프로토콜별 파싱(AVTP/gPTP/CAN-UDP) 취약성을 피하는 견고한 선택이다.
필요 시 strip_l2=True 로 Ethernet 14B 헤더를 제거할 수 있다(튜닝 포인트).
"""
from __future__ import annotations

from pathlib import Path

import dpkt

ETH_HEADER_LEN = 14


def parse_pcap(
    path: str | Path,
    strip_l2: bool = False,
    max_packets: int | None = None,
) -> list[bytes]:
    """pcap/pcapng → 패킷별 raw 바이트 리스트."""
    if max_packets is not None and max_packets < 0:
        raise ValueError(f'max_packets must be non-negative, got {max_packets}')
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
                frames.append(frame[ETH_HEADER_LEN:] if strip_l2 else frame)
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
