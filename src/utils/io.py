"""dataset.npz 계약 (KEY CONTRACT) — 팀 전체의 유일한 공유 결합점.

규격 (PLAN.md):
    X : np.float32, shape (N, 3, H, W), 값 범위 [0, 1]   (채널 = coif1, db3, rbio1.3 의 LL)
    y : np.int64,  shape (N,),         값 {0..5}
    meta : dict (npz 내부에 JSON으로 동봉)

라벨 맵 (전원 동일):
    0=Normal, 1=F_I(Frame Injection), 2=P_I(PTP Sync),
    3=M_F(Switch/MAC Flooding), 4=C_D(CAN DoS), 5=C_R(CAN Replay)

이 모듈만이 npz를 읽고/쓰며, save 시 자동으로 validate_schema 를 통과해야 한다.
스키마를 바꿀 때만 팀 합의가 필요하다.
"""
from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np

# --- 라벨 계약 -------------------------------------------------------------
LABEL_MAP: dict[str, int] = {
    "Normal": 0,
    "F_I": 1,  # Frame Injection (AVTP)
    "P_I": 2,  # PTP Sync (gPTP)
    "M_F": 3,  # Switch / MAC Flooding
    "C_D": 4,  # CAN DoS
    "C_R": 5,  # CAN Replay
}
LABEL_NAMES: list[str] = list(LABEL_MAP)
NUM_CLASSES = len(LABEL_MAP)  # 6 (학습 클래스). Unknown(6)은 추론 시점 결정.
UNKNOWN_LABEL = NUM_CLASSES  # 6 — 2단계 파이프라인 최종 출력에서만 사용
N_CHANNELS = 3
DATASET_SCHEMA_VERSION = 2
CANONICAL_WAVELETS = ["coif1", "db3", "rbio1.3"]

_EPS = 1e-5


class SchemaError(ValueError):
    """dataset.npz 계약 위반."""


def validate_schema(X: np.ndarray, y: np.ndarray, meta: dict | None = None,
                    pcap_id: np.ndarray | None = None,
                    packet_start: np.ndarray | None = None,
                    packet_end: np.ndarray | None = None) -> None:
    """계약 위반 시 SchemaError 를 raise. 통과하면 None."""
    errs: list[str] = []

    # --- X ---
    if not isinstance(X, np.ndarray):
        errs.append(f"X must be np.ndarray, got {type(X)}")
    else:
        if X.dtype != np.float32:
            errs.append(f"X.dtype must be float32, got {X.dtype}")
        if X.ndim != 4 or X.shape[1] != N_CHANNELS:
            errs.append(f"X.shape must be (N,{N_CHANNELS},H,W), got {X.shape}")
        if X.size:
            if not np.isfinite(X).all():
                errs.append("X has non-finite values (nan/inf)")
            lo, hi = float(np.nanmin(X)), float(np.nanmax(X))
            if lo < -_EPS or hi > 1 + _EPS:
                errs.append(f"X out of [0,1] range: min={lo:.4g}, max={hi:.4g}")

    # --- y ---
    if not isinstance(y, np.ndarray):
        errs.append(f"y must be np.ndarray, got {type(y)}")
    else:
        if y.dtype != np.int64:
            errs.append(f"y.dtype must be int64, got {y.dtype}")
        if y.ndim != 1:
            errs.append(f"y must be 1-D, got shape {y.shape}")
        if (y.ndim == 1 and isinstance(X, np.ndarray) and X.ndim == 4
                and len(y) != len(X)):
            errs.append(f"len(y)={len(y)} != len(X)={len(X)}")
        if y.size and (int(y.min()) < 0 or int(y.max()) > NUM_CLASSES - 1):
            errs.append(f"y out of label range {{0..{NUM_CLASSES-1}}}: "
                        f"min={int(y.min())}, max={int(y.max())}")

    # --- meta (선택) ---
    if meta is not None and not isinstance(meta, Mapping):
        errs.append(f"meta must be a mapping, got {type(meta)}")
    elif meta is not None:
        required = {
            "schema_version", "wavelets", "H", "W", "mode", "norm",
            "label_map", "seed", "git_sha", "created_at",
            "packets_per_image", "bytes_per_packet", "stride",
            "frame_content", "partial_window",
        }
        missing = required - set(meta)
        if missing:
            errs.append(f"meta missing keys: {sorted(missing)}")
        if meta.get("label_map") not in (None, LABEL_MAP):
            errs.append("meta.label_map differs from the canonical LABEL_MAP")
        if meta.get("schema_version") != DATASET_SCHEMA_VERSION:
            errs.append(
                f"meta.schema_version must be {DATASET_SCHEMA_VERSION}, "
                f"got {meta.get('schema_version')}")
        if meta.get("wavelets") != CANONICAL_WAVELETS:
            errs.append(
                f"meta.wavelets must be {CANONICAL_WAVELETS}, "
                f"got {meta.get('wavelets')}")
        if meta.get("mode") != "periodization":
            errs.append(f"meta.mode must be periodization, got {meta.get('mode')}")
        if meta.get("norm") != "minmax01":
            errs.append(f"meta.norm must be minmax01, got {meta.get('norm')}")
        if meta.get("frame_content") not in {"payload", "full_frame"}:
            errs.append("meta.frame_content must be payload or full_frame")
        if meta.get("partial_window") not in {"pad", "drop"}:
            errs.append("meta.partial_window must be pad or drop")
        for key in ("packets_per_image", "bytes_per_packet", "stride"):
            try:
                if int(meta.get(key, 0)) <= 0:
                    errs.append(f"meta.{key} must be a positive integer")
            except (TypeError, ValueError):
                errs.append(f"meta.{key} must be a positive integer")
        # meta H/W 가 실제 X 공간 차원과 일치하는지 교차검증
        if isinstance(X, np.ndarray) and X.ndim == 4 and "H" in meta and "W" in meta:
            try:
                meta_hw = int(meta["H"]), int(meta["W"])
            except (TypeError, ValueError):
                errs.append(f"meta H/W must be integers, got ({meta['H']},{meta['W']})")
            else:
                if meta_hw != (X.shape[2], X.shape[3]):
                    errs.append(f"meta H/W ({meta['H']},{meta['W']}) != X spatial "
                                f"({X.shape[2]},{X.shape[3]})")

    # --- pcap_id (선택, 누수 안전 split 용) ---
    if pcap_id is not None:
        if not isinstance(pcap_id, np.ndarray) or pcap_id.dtype != np.int64:
            errs.append(f"pcap_id must be int64 ndarray, got "
                        f"{getattr(pcap_id, 'dtype', type(pcap_id))}")
        elif pcap_id.ndim != 1:
            errs.append(f"pcap_id must be 1-D, got shape {pcap_id.shape}")
        elif isinstance(X, np.ndarray) and X.ndim == 4 and len(pcap_id) != len(X):
            errs.append(f"len(pcap_id)={len(pcap_id)} != len(X)={len(X)}")
        if meta is None or meta.get("group_semantics") != "capture":
            errs.append("pcap_id requires meta.group_semantics=capture")
        if packet_start is None or packet_end is None:
            errs.append("pcap_id requires packet_start and packet_end provenance")

    # --- packet provenance (선택, [start, end) 범위) ---
    if (packet_start is None) != (packet_end is None):
        errs.append('packet_start and packet_end must be provided together')
    elif packet_start is not None and packet_end is not None:
        for name, values in (('packet_start', packet_start), ('packet_end', packet_end)):
            if not isinstance(values, np.ndarray) or values.dtype != np.int64:
                errs.append(f'{name} must be int64 ndarray, got '
                            f'{getattr(values, "dtype", type(values))}')
            elif values.ndim != 1:
                errs.append(f'{name} must be 1-D, got shape {values.shape}')
            elif isinstance(X, np.ndarray) and X.ndim == 4 and len(values) != len(X):
                errs.append(f'len({name})={len(values)} != len(X)={len(X)}')
        if (isinstance(packet_start, np.ndarray) and packet_start.dtype == np.int64
                and packet_start.ndim == 1
                and isinstance(packet_end, np.ndarray) and packet_end.dtype == np.int64
                and packet_end.ndim == 1 and len(packet_start) == len(packet_end)):
            if np.any(packet_start < 0):
                errs.append('packet_start must be non-negative')
            if np.any(packet_end <= packet_start):
                errs.append('packet_end must be greater than packet_start')
            if pcap_id is not None and isinstance(pcap_id, np.ndarray):
                for capture_id in np.unique(pcap_id):
                    capture_idx = np.flatnonzero(pcap_id == capture_id)
                    starts = packet_start[capture_idx]
                    if np.any(starts[1:] < starts[:-1]):
                        errs.append(
                            f'packet_start must be monotonic within capture {int(capture_id)}')

    if errs:
        raise SchemaError("dataset.npz 계약 위반:\n  - " + "\n  - ".join(errs))


def build_meta(H: int, W: int, *, wavelets=("coif1", "db3", "rbio1.3"),
               mode: str = "periodization", norm: str = "minmax01",
               seed: int | None = None, git_sha: str | None = None,
               created_at: str | None = None,
               packets_per_image: int = 64, bytes_per_packet: int = 64,
               stride: int = 64, frame_content: str = "payload",
               partial_window: str = "pad", **extra) -> dict:
    """계약을 만족하는 meta dict 생성."""
    meta = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "wavelets": list(wavelets),
        "H": int(H),
        "W": int(W),
        "mode": mode,
        "norm": norm,
        "label_map": dict(LABEL_MAP),
        "seed": seed,
        "git_sha": git_sha,
        "created_at": created_at,
        "packets_per_image": int(packets_per_image),
        "bytes_per_packet": int(bytes_per_packet),
        "stride": int(stride),
        "frame_content": frame_content,
        "partial_window": partial_window,
    }
    meta.update(extra)
    return meta


def _json_default(o):
    """JSON encoder for NumPy metadata values."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for an artifact."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_dataset(path: str | Path, X: np.ndarray, y: np.ndarray,
                 meta: dict | None = None, pcap_id: np.ndarray | None = None,
                 packet_start: np.ndarray | None = None,
                 packet_end: np.ndarray | None = None) -> Path:
    """검증 후 npz 저장. meta 는 JSON(uint8 byte array)으로 동봉(pickle 불필요).

    pcap_id (선택): (N,) int64 — 각 샘플이 온 capture(PCAP) id. 누수 안전 split 용.
    저장 시 사람이 읽을 수 있는 `<path>.meta.json` sidecar 도 함께 작성한다.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if not np.issubdtype(X.dtype, np.floating):
        raise SchemaError(f'X must have a floating dtype before storage, got {X.dtype}')
    if not np.issubdtype(y.dtype, np.integer):
        raise SchemaError(f'y must have an integer dtype before storage, got {y.dtype}')
    if pcap_id is not None:
        pcap_id = np.asarray(pcap_id)
        if not np.issubdtype(pcap_id.dtype, np.integer):
            raise SchemaError(
                f'pcap_id must have an integer dtype before storage, got {pcap_id.dtype}')
    if packet_start is not None:
        packet_start = np.asarray(packet_start)
        if not np.issubdtype(packet_start.dtype, np.integer):
            raise SchemaError('packet_start must have an integer dtype before storage')
    if packet_end is not None:
        packet_end = np.asarray(packet_end)
        if not np.issubdtype(packet_end.dtype, np.integer):
            raise SchemaError('packet_end must have an integer dtype before storage')

    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.int64)
    if pcap_id is not None:
        pcap_id = np.ascontiguousarray(pcap_id, dtype=np.int64)
    if packet_start is not None:
        packet_start = np.ascontiguousarray(packet_start, dtype=np.int64)
    if packet_end is not None:
        packet_end = np.ascontiguousarray(packet_end, dtype=np.int64)
    if meta is None:
        validate_schema(X, y, pcap_id=pcap_id,
                        packet_start=packet_start, packet_end=packet_end)
        meta = build_meta(X.shape[2], X.shape[3])
    validate_schema(X, y, meta, pcap_id, packet_start, packet_end)

    path = Path(path)
    out = path if path.suffix == ".npz" else Path(f"{path}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_str = json.dumps(meta, default=_json_default)
    meta_bytes = np.frombuffer(meta_str.encode("utf-8"), dtype=np.uint8)

    arrays = {"X": X, "y": y, "meta_json": meta_bytes}
    if pcap_id is not None:
        arrays["pcap_id"] = pcap_id
    if packet_start is not None:
        arrays['packet_start'] = packet_start
        arrays['packet_end'] = packet_end
    np.savez_compressed(out, **arrays)
    # 사람이 읽을 수 있는 meta.json sidecar (권호영 스펙)
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, default=_json_default, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return out


def load_dataset(path: str | Path, with_pcap_id: bool = False):
    """npz 로드 후 계약 재검증.

    기본: (X, y, meta) 반환 (하위호환).
    with_pcap_id=True: (X, y, meta, pcap_id) 반환 (없으면 pcap_id=None).
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as npz:
            X = np.asarray(npz["X"])
            y = np.asarray(npz["y"])
            meta = {}
            if "meta_json" in npz:
                meta = json.loads(bytes(npz["meta_json"]).decode("utf-8"))
            pcap_id = (np.asarray(npz["pcap_id"])
                       if "pcap_id" in npz else None)
            packet_start = (np.asarray(npz['packet_start'])
                            if 'packet_start' in npz else None)
            packet_end = (np.asarray(npz['packet_end'])
                          if 'packet_end' in npz else None)
    except (OSError, KeyError, ValueError, UnicodeDecodeError,
            json.JSONDecodeError) as exc:
        raise SchemaError(f"failed to load dataset {path}: {exc}") from exc
    validate_schema(X, y, meta or None, pcap_id, packet_start, packet_end)
    sidecar_path = path.with_suffix('.meta.json')
    try:
        sidecar_meta = json.loads(sidecar_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f'failed to load metadata sidecar {sidecar_path}: {exc}') from exc
    if sidecar_meta != meta:
        raise SchemaError('embedded metadata does not match the metadata sidecar')
    if with_pcap_id:
        return X, y, meta, pcap_id
    return X, y, meta


def load_provenance(path: str | Path) -> tuple[np.ndarray | None, np.ndarray | None,
                                                np.ndarray | None]:
    """Load capture IDs and packet ranges without loading the image tensor."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as npz:
            pcap_id = np.asarray(npz['pcap_id']) if 'pcap_id' in npz else None
            packet_start = (np.asarray(npz['packet_start'])
                            if 'packet_start' in npz else None)
            packet_end = np.asarray(npz['packet_end']) if 'packet_end' in npz else None
    except (OSError, ValueError) as exc:
        raise SchemaError(f'failed to load provenance from {path}: {exc}') from exc
    if pcap_id is not None and (pcap_id.dtype != np.int64 or pcap_id.ndim != 1):
        raise SchemaError('pcap_id must be a 1-D int64 array')
    for name, values in (('packet_start', packet_start), ('packet_end', packet_end)):
        if values is not None and (values.dtype != np.int64 or values.ndim != 1):
            raise SchemaError(f'{name} must be a 1-D int64 array')
    if (packet_start is None) != (packet_end is None):
        raise SchemaError('packet_start and packet_end must be provided together')
    return pcap_id, packet_start, packet_end


def load_pcap_id(path: str | Path) -> np.ndarray | None:
    """npz 에서 pcap_id 만 로드 (없으면 None)."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as npz:
            pcap_id = (np.asarray(npz["pcap_id"])
                       if "pcap_id" in npz else None)
        if pcap_id is not None and (pcap_id.dtype != np.int64 or pcap_id.ndim != 1):
            raise SchemaError(
                f'pcap_id must be a 1-D int64 array, got {pcap_id.shape}/{pcap_id.dtype}')
        return pcap_id
    except (OSError, ValueError) as exc:
        raise SchemaError(f"failed to load pcap_id from {path}: {exc}") from exc
