from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(text, encoding="utf-8")
    partial.chmod(0o600)
    partial.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def source_fingerprint(path: Path, sample_bytes: int = 1024 * 1024) -> str:
    """Hash metadata plus the beginning/end of a file without reading a huge video."""
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    digest = hashlib.sha256()
    digest.update(str(resolved).encode("utf-8", errors="surrogateescape"))
    digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    with resolved.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if stat.st_size > sample_bytes:
            handle.seek(max(0, stat.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def safe_stem(path: Path) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)
    return cleaned.strip("_") or "video"


def user_output_prefix(source: str | Path) -> str:
    """Return a stable, download-friendly prefix derived from the source name."""
    source_path = Path(source)
    date_match = re.search(r"(?<!\d)(20\d{6})(?!\d)", source_path.name)
    if date_match:
        return f"Robotics_Seminar_{date_match.group(1)}"
    return safe_stem(source_path)
