from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from app.core.config import settings
from app.core.logger import log


from app.transcribe.utlities.evaluation_engine import load_scheme, SchemeDef


@dataclass(frozen=True)
class SchemeMeta:
    system_code: str
    version: str
    valid_from: date
    valid_to: Optional[date]
    scheme_path: Path


def _parse_date(s: str) -> date:
    # expects YYYY-MM-DD
    return datetime.strptime(s, "%Y-%m-%d").date()


def _load_scheme_meta(scheme_path: Path) -> SchemeMeta:
    raw = json.loads(scheme_path.read_text(encoding="utf-8"))

    system_code = raw["system_code"]
    version = raw.get("version", raw.get("ver", "0.0.0"))

    vf = _parse_date(raw["valid_from"])
    vt_raw = raw.get("valid_to")
    vt = _parse_date(vt_raw) if vt_raw else None

    return SchemeMeta(
        system_code=system_code,
        version=version,
        valid_from=vf,
        valid_to=vt,
        scheme_path=scheme_path,
    )

def _version_key(v: str):
    """
    Comparable key for versions like: 1.2.10, 2.0, 1.0.0-beta
    Numeric parts sort numerically, non-numeric lexicographically.
    """
    parts = re.split(r"[.\-_+]", (v or "").strip())
    key = []
    for p in parts:
        if p == "":
            continue
        if p.isdigit():
            key.append((0, int(p)))
        else:
            key.append((1, p.lower()))
    return tuple(key)


def list_schemes_for_system(config_root: str | Path, system_code: str) -> List[SchemeMeta]:
    """
    Scans: <config_root>/<system_code>/**/scheme.json
    """
    root = Path(config_root).resolve()
    base = root / system_code
    if not base.exists():
        return []

    metas: List[SchemeMeta] = []
    for scheme_path in base.rglob("scheme.json"):
        metas.append(_load_scheme_meta(scheme_path))

    return metas




def resolve_scheme_path(
    config_root: str | Path,
    system_code: str,
    call_date: date,
    version: Optional[str] = None,
) -> Path:
    metas = list_schemes_for_system(config_root, system_code)

    if not metas:
        raise FileNotFoundError(f"No scheme.json found for system_code={system_code}")

    if version:
        for m in metas:
            if m.version == version:
                return m.scheme_path
        raise FileNotFoundError(f"No scheme.json for system_code={system_code} version={version}")

    # pick schemes whose validity window contains call_date
    candidates = []
    for m in metas:
        if m.valid_from <= call_date and (m.valid_to is None or call_date <= m.valid_to):
            candidates.append(m)

    if not candidates:
        raise ValueError(f"No active scheme for system_code={system_code} on {call_date.isoformat()}")

    # NEW: pick latest valid_from, then highest version within that valid_from
    latest_vf = max(m.valid_from for m in candidates)
    latest_candidates = [m for m in candidates if m.valid_from == latest_vf]
    latest_candidates.sort(key=lambda m: _version_key(m.version), reverse=True)
    return latest_candidates[0].scheme_path



def _try_extract_scheme_ref_from_result(result: Optional[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    """
    If prev_result contains the scheme reference, return (system_code, scheme_version).
    Otherwise return None.
    """
    if not result or not isinstance(result, dict):
        return None

    system_code = (result.get("system_code") or "").strip()
    version = (result.get("scheme_version") or result.get("version") or "").strip()

    if system_code and version:
        return system_code, version

    return None




def load_active_scheme(
    config_root,
    system_code: str,
    call_date: date,
    call_info: Optional[Dict[str, Any]] = None,
    prev_result: Optional[Dict[str, Any]] = None,
):
    """
    Priority:
      1) prev_result (if it contains system_code + scheme_version) -> load exact scheme version
      2) otherwise -> load by validity window using (system_code, call_date)
    """
    # 1) Prev_result is priority (if usable)
    ref = _try_extract_scheme_ref_from_result(prev_result)
    if ref is not None:
        prev_sys_code, prev_version = ref
        scheme_path = resolve_scheme_path(
            config_root=config_root,
            system_code=prev_sys_code,
            call_date=None,
            version=prev_version,
        )
        return load_scheme(scheme_path, call_info=call_info)

    # 2) Fallback: select by date
    scheme_path = resolve_scheme_path(
        config_root=config_root,
        system_code=system_code,
        call_date=call_date,
        version=None,
    )
    return load_scheme(scheme_path, call_info=call_info)