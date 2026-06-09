"""전처리(wavelet/imaging/pcap) 테스트 — 합성 pcap 으로 실데이터 없이 검증."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.imaging import (  # noqa: E402
    frames_to_images, payload_to_row, window_packet_ranges,
)
from src.preprocessing.pcap_parser import parse_pcap, write_pcap  # noqa: E402
from src.preprocessing.wavelet import output_hw, wavelet_3ch  # noqa: E402
from src.utils.io import validate_schema  # noqa: E402


def test_wavelet_shape_and_range():
    img = np.random.rand(64, 64).astype(np.float32)
    out = wavelet_3ch(img, level=1)
    assert out.shape == (3, 32, 32), out.shape          # dwt2 가 차원 반감
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0         # 계약 [0,1]


def test_wavelet_rbio_alignment():
    # rbio1.3(biorthogonal) 포함해도 3채널 stack 이 정렬되는지
    for size in (32, 60, 64, 116):
        out = wavelet_3ch(np.random.rand(size, size).astype(np.float32))
        assert out.shape[0] == 3 and out.shape[1] == out.shape[2]


def test_output_hw_matches_stub():
    assert output_hw(64, level=1) == (32, 32)


def test_payload_pad_truncate():
    assert payload_to_row(b"\x01\x02", 4).tolist() == [1, 2, 0, 0]
    assert payload_to_row(b"\x01\x02\x03\x04\x05", 3).tolist() == [1, 2, 3]


def test_invalid_imaging_parameters_raise():
    with pytest.raises(ValueError, match='stride'):
        frames_to_images(
            [b'abc'], [0], packets_per_image=1, bytes_per_packet=3, stride=0)
    with pytest.raises(ValueError, match='n_bytes'):
        payload_to_row(b'abc', 0)


def test_pcap_roundtrip_and_imaging(tmp_path):
    frames = [(np.arange(80) + i).astype(np.uint8).tobytes() for i in range(200)]
    p = tmp_path / "syn.pcap"
    write_pcap(p, frames)
    parsed = parse_pcap(p, frame_content='full_frame')
    assert len(parsed) == 200
    labels = np.full(200, 4, dtype=np.int64)  # CAN DoS
    imgs, ilabels, igroups = frames_to_images(
        parsed, labels, packets_per_image=64, bytes_per_packet=64, stride=64, group_id=7)
    assert imgs.shape == (4, 64, 64) and imgs.max() <= 1.0
    assert (ilabels == 4).all() and (igroups == 7).all()
    starts, ends = window_packet_ranges(200, 64, 64, pad_partial=True)
    assert starts.tolist() == [0, 64, 128, 192]
    assert ends.tolist() == [64, 128, 192, 200]


def test_full_pipeline_to_contract(tmp_path):
    # 합성 pcap → 이미징 → wavelet → dataset.npz 계약 통과
    frames = [bytes(np.random.randint(0, 256, 90, np.uint8).tobytes()) for _ in range(130)]
    p = tmp_path / "s.pcap"; write_pcap(p, frames)
    imgs, ilabels, _ = frames_to_images(
        parse_pcap(p), np.zeros(130, np.int64),
        packets_per_image=64, bytes_per_packet=64, stride=64, group_id=0)
    X = np.stack([wavelet_3ch(im) for im in imgs]).astype(np.float32)
    validate_schema(X, ilabels)
