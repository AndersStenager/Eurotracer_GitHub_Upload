"""Ordinary files, atomic checkpoints, and bounded reads. No object pool."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
import zarr

FORMAT = 3


def plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def identity(value):
    return hashlib.sha256(
        json.dumps(plain(value), sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()[:24]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(plain(value), f, indent=2, sort_keys=True, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


@contextmanager
def lock(path):
    """OS lock is released on process exit, including an interrupted run."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a+b") as f:
        f.seek(0)
        if not f.read(1):
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl = import_module("fcntl")

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process is writing {path}") from exc
        try:
            yield path
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def snapshot(source, destination, names):
    """Copy small source inputs once; leave expensive archives in place."""
    hashes = {
        name: file_hash(Path(source) / name) for name in names if (Path(source) / name).is_file()
    }
    key = identity(hashes)
    target = Path(destination) / key
    with lock(target):
        if not (target / "snapshot.json").exists():
            for name, expected in hashes.items():
                shutil.copyfile(Path(source) / name, target / name)
                if file_hash(target / name) != expected:
                    raise RuntimeError(f"Input changed while snapshotting {name}")
            write_json(target / "snapshot.json", hashes)
    return target, hashes


def group(path, mode="r"):
    return zarr.open_group(str(path), mode=mode, zarr_format=2)


def array(root, name, shape, *, dtype="f8", chunks=None):
    """Create once; repeated writes overwrite only uncommitted regions."""
    if name in root:
        result = root[name]
        if result.shape != tuple(shape) or np.dtype(result.dtype) != np.dtype(dtype):
            raise ValueError(f"Stored shape/dtype mismatch for {name}")
        return result
    if chunks is None:
        chunks = (min(168, max(1, shape[0])), *[max(1, min(n, 256)) for n in shape[1:]])
    return root.create_array(name, shape=tuple(shape), dtype=dtype, chunks=chunks, fill_value=0)


def slices(count, size=168):
    for start in range(0, count, size):
        yield slice(start, min(start + size, count))


def quantile_columns(values, q, *, absolute=False, negative=False, memory_gib=24):
    """Exact NumPy linear quantiles; memory grows with selected columns, not years."""
    hours, columns = values.shape
    width = max(1, min(columns, int(memory_gib * 1024**3 * 0.1) // max(1, hours * 24)))
    answer = np.zeros(columns)
    for block in slices(columns, width):
        x = np.asarray(values[:, block], dtype=np.float64)
        if absolute:
            np.abs(x, out=x)
        if negative:
            np.negative(x, out=x)
            np.maximum(x, 0, out=x)
        answer[block] = np.quantile(x, q, axis=0, method="linear")
    return answer


def safe_name(value):
    value = str(value)
    if not value or value in (".", "..") or any(c in value for c in "/\\:"):
        raise ValueError("Names must be a single nonempty directory component")
    return value
