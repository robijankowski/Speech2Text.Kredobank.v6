from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.logger import log

# --- optional JSONSchema validation (if installed) ---
try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None


@dataclass
class Issue:
    level: str            # "ERROR" | "WARN"
    code: str             # machine-friendly code
    message: str          # human-friendly
    path: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class SchemeAuditReport:
    scheme_path: str
    system_code: Optional[str]
    version: Optional[str]
    valid_from: Optional[str]
    valid_to: Optional[str]
    default_model: Optional[str]
    checks_count: int
    errors: List[Issue]
    warnings: List[Issue]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_date_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _add_issue(issues: List[Issue], level: str, code: str, msg: str, *, path: Optional[Path] = None, context: Optional[Dict[str, Any]] = None):
    issues.append(Issue(level=level, code=code, message=msg, path=str(path) if path else None, context=context))


def _validate_score_schema_vs_check(
    *,
    schema: Dict[str, Any],
    check_id: str,
    check_path: Path,
    max_points: int,
    result_key: str,
    errors: List[Issue],
    warnings: List[Issue],
) -> None:
    # Minimal sanity checks
    if schema.get("type") != "object":
        _add_issue(errors, "ERROR", "schema.type", f"Schema type should be 'object' for check '{check_id}'.", path=check_path)
        return

    props = schema.get("properties")
    if not isinstance(props, dict):
        _add_issue(errors, "ERROR", "schema.properties", f"Schema has no 'properties' object for check '{check_id}'.", path=check_path)
        return

    if result_key not in props:
        _add_issue(errors, "ERROR", "schema.result_key_missing", f"Schema properties missing result_key='{result_key}' for check '{check_id}'.", path=check_path)
        return

    rk = props.get(result_key, {})
    if not isinstance(rk, dict):
        _add_issue(errors, "ERROR", "schema.result_key_invalid", f"Schema property '{result_key}' must be an object for check '{check_id}'.", path=check_path)
        return

    rk_type = rk.get("type")
    if rk_type != "integer":
        _add_issue(warnings, "WARN", "schema.score_not_integer", f"Schema '{result_key}.type' is '{rk_type}', expected 'integer' for check '{check_id}'.", path=check_path)

    # Compare maximum if present
    schema_max = rk.get("maximum")
    if isinstance(schema_max, int) and schema_max != max_points:
        _add_issue(
            warnings,
            "WARN",
            "schema.max_mismatch",
            f"Schema maximum ({schema_max}) != check.max_points ({max_points}) for check '{check_id}'.",
            path=check_path,
            context={"schema_max": schema_max, "check_max_points": max_points},
        )

    # Recommended strictness
    if schema.get("additionalProperties") is not False:
        _add_issue(warnings, "WARN", "schema.additionalProperties",
                   f"Recommended: additionalProperties=false for strict outputs (check '{check_id}').", path=check_path)

    req = schema.get("required")
    if isinstance(req, list) and result_key not in req:
        _add_issue(warnings, "WARN", "schema.required_missing",
                   f"Recommended: include '{result_key}' in schema.required for check '{check_id}'.", path=check_path)


def audit_evaluation_configs(config_root: str | Path) -> List[Dict[str, Any]]:
    """
    Skanuje wszystkie scheme.json pod config_root i tworzy raport spójności dla każdej konfiguracji.

    Nowe:
      - obsługa scheme.json["conditional_checks"] :
        format: { "<TAG>": ["checks/a.json", "checks/b.json", ...], ... }
      - audyt obejmuje checks bazowe + wszystkie conditional_checks (deduplikowane)
    """
    root = Path(config_root).resolve()
    scheme_files = sorted(root.rglob("scheme.json"))

    reports: List[SchemeAuditReport] = []

    for scheme_path in scheme_files:
        errors: List[Issue] = []
        warnings: List[Issue] = []

        scheme_raw: Dict[str, Any] = {}
        try:
            scheme_raw = _read_json(scheme_path)
        except Exception as e:
            _add_issue(errors, "ERROR", "scheme.read_failed", f"Cannot read/parse scheme.json: {e}", path=scheme_path)
            reports.append(SchemeAuditReport(
                scheme_path=str(scheme_path),
                system_code=None, version=None, valid_from=None, valid_to=None, default_model=None,
                checks_count=0, errors=errors, warnings=warnings
            ))
            continue

        system_code = scheme_raw.get("system_code")
        version = scheme_raw.get("version") or scheme_raw.get("ver")
        valid_from = scheme_raw.get("valid_from")
        valid_to = scheme_raw.get("valid_to")
        default_model = scheme_raw.get("default_model")

        # --- base checks ---
        checks = scheme_raw.get("checks", [])

        # --- NEW: conditional checks ---
        conditional_map = scheme_raw.get("conditional_checks")

        # Validate conditional_checks structure
        if conditional_map is not None and not isinstance(conditional_map, dict):
            _add_issue(
                warnings, "WARN", "scheme.conditional_checks_type",
                "'conditional_checks' should be an object/dict mapping tag -> list of check paths.",
                path=scheme_path
            )
            conditional_map = {}

        # Collect all check paths (base + conditional), dedup while preserving order
        all_check_paths: List[str] = []
        seen_paths: set[str] = set()

        def _add_check_path(p: Any, *, where: str):
            if not isinstance(p, str) or not p.strip():
                _add_issue(
                    warnings, "WARN", "scheme.check_path_invalid",
                    f"Invalid check path in {where}: {p!r}",
                    path=scheme_path,
                )
                return
            if p not in seen_paths:
                seen_paths.add(p)
                all_check_paths.append(p)

        # Add base checks
        if isinstance(checks, list):
            for rel in checks:
                _add_check_path(rel, where="checks")
        else:
            _add_issue(errors, "ERROR", "scheme.checks", "Missing/invalid 'checks' list in scheme.json.", path=scheme_path)

        # Add conditional checks
        if isinstance(conditional_map, dict):
            for tag, paths in conditional_map.items():
                if not isinstance(tag, str) or not tag.strip():
                    _add_issue(
                        warnings, "WARN", "scheme.conditional_checks_key_invalid",
                        f"Invalid conditional_checks key (expected non-empty string): {tag!r}",
                        path=scheme_path,
                    )
                    continue
                if not isinstance(paths, list):
                    _add_issue(
                        warnings, "WARN", "scheme.conditional_checks_value_type",
                        f"conditional_checks['{tag}'] should be a list of check paths.",
                        path=scheme_path,
                    )
                    continue
                for rel in paths:
                    _add_check_path(rel, where=f"conditional_checks['{tag}']")

        # --- scheme-level validation (existing) ---
        if not isinstance(system_code, str) or not system_code.strip():
            _add_issue(errors, "ERROR", "scheme.system_code", "Missing/invalid 'system_code' in scheme.json.", path=scheme_path)

        if not isinstance(version, str) or not version.strip():
            _add_issue(warnings, "WARN", "scheme.version", "Missing/invalid 'version' in scheme.json (will fall back to 'ver'/'0.0.0' in loader).", path=scheme_path)

        vf_parsed = None
        vt_parsed = None
        if not isinstance(valid_from, str) or not valid_from.strip():
            _add_issue(errors, "ERROR", "scheme.valid_from", "Missing/invalid 'valid_from' (expected YYYY-MM-DD).", path=scheme_path)
        else:
            try:
                vf_parsed = _parse_date_yyyy_mm_dd(valid_from)
            except Exception as e:
                _add_issue(errors, "ERROR", "scheme.valid_from_parse", f"Cannot parse valid_from='{valid_from}': {e}", path=scheme_path)

        if valid_to is not None:
            if not isinstance(valid_to, str) or not valid_to.strip():
                _add_issue(errors, "ERROR", "scheme.valid_to", "Invalid 'valid_to' (expected null or YYYY-MM-DD).", path=scheme_path)
            else:
                try:
                    vt_parsed = _parse_date_yyyy_mm_dd(valid_to)
                except Exception as e:
                    _add_issue(errors, "ERROR", "scheme.valid_to_parse", f"Cannot parse valid_to='{valid_to}': {e}", path=scheme_path)

        if vf_parsed and vt_parsed and vt_parsed < vf_parsed:
            _add_issue(errors, "ERROR", "scheme.valid_window", "valid_to is earlier than valid_from.", path=scheme_path,
                       context={"valid_from": valid_from, "valid_to": valid_to})

        if not isinstance(default_model, str) or not default_model.strip():
            _add_issue(warnings, "WARN", "scheme.default_model", "Missing/invalid 'default_model' in scheme.json.", path=scheme_path)

        # If after combining there are no checks at all, error
        if not all_check_paths:
            _add_issue(errors, "ERROR", "scheme.checks_empty", "No checks found (checks + conditional_checks are empty).", path=scheme_path)

        scoring = scheme_raw.get("scoring", {})
        if scoring and not isinstance(scoring, dict):
            _add_issue(warnings, "WARN", "scheme.scoring_type", "'scoring' should be an object.", path=scheme_path)
        if isinstance(scoring, dict):
            agg = scoring.get("aggregation")
            if agg and agg not in {"weighted_sum"}:
                _add_issue(warnings, "WARN", "scheme.aggregation_unknown", f"Unknown scoring.aggregation='{agg}'.", path=scheme_path)

        # --- per-check validation (UPDATED: iterate over all_check_paths) ---
        seen_ids: set[str] = set()
        scheme_dir = scheme_path.parent

        for rel_check_path in all_check_paths:
            check_file = (scheme_dir / str(rel_check_path)).resolve()
            if not check_file.exists():
                _add_issue(errors, "ERROR", "check.file_missing", f"Check file not found: {rel_check_path}", path=check_file)
                continue

            try:
                chk = _read_json(check_file)
            except Exception as e:
                _add_issue(errors, "ERROR", "check.read_failed", f"Cannot read/parse check json: {e}", path=check_file)
                continue

            check_id = chk.get("id")
            if not isinstance(check_id, str) or not check_id.strip():
                _add_issue(errors, "ERROR", "check.id", "Missing/invalid check 'id'.", path=check_file)
                continue

            if check_id in seen_ids:
                _add_issue(errors, "ERROR", "check.id_duplicate", f"Duplicate check id='{check_id}' in scheme.", path=check_file)
            seen_ids.add(check_id)

            # ... keep the rest of your existing per-check validation unchanged ...
            # (desc, weight, max_points, temperature, prompt files, schema file, schema validation)

        reports.append(SchemeAuditReport(
            scheme_path=str(scheme_path),
            system_code=system_code if isinstance(system_code, str) else None,
            version=version if isinstance(version, str) else None,
            valid_from=valid_from if isinstance(valid_from, str) else None,
            valid_to=valid_to if isinstance(valid_to, str) else None,
            default_model=default_model if isinstance(default_model, str) else None,
            checks_count=len(all_check_paths),
            errors=errors,
            warnings=warnings,
        ))

    return [
        {
            **{k: v for k, v in asdict(r).items() if k not in {"errors", "warnings"}},
            "ok": r.ok,
            "errors": [asdict(i) for i in r.errors],
            "warnings": [asdict(i) for i in r.warnings],
        }
        for r in reports
    ]




def format_audit_report_md(reports: List[Dict[str, Any]]) -> str:
    """
    Proste formatowanie raportu do Markdown (do logów / README / artefaktu).
    """
    lines: List[str] = []
    total = len(reports)
    ok = sum(1 for r in reports if r.get("ok"))
    bad = total - ok

    lines.append(f"# Audit evaluation configs")
    lines.append(f"- Total schemes: **{total}**")
    lines.append(f"- OK: **{ok}**")
    lines.append(f"- With errors: **{bad}**")
    lines.append("")

    for r in reports:
        header = f"## {r.get('system_code') or '???'} v{r.get('version') or '???'}"
        lines.append(header)
        lines.append(f"- Path: `{r.get('scheme_path')}`")
        lines.append(f"- Validity: `{r.get('valid_from')}` → `{r.get('valid_to')}`")
        lines.append(f"- Default model: `{r.get('default_model')}`")
        lines.append(f"- Checks: `{r.get('checks_count')}`")
        lines.append(f"- Status: {'✅ OK' if r.get('ok') else '❌ ERRORS'}")
        lines.append("")

        errs = r.get("errors") or []
        warns = r.get("warnings") or []

        if errs:
            lines.append("### Errors")
            for e in errs:
                lines.append(f"- **{e.get('code')}**: {e.get('message')}" + (f" (`{e.get('path')}`)" if e.get("path") else ""))
            lines.append("")

        if warns:
            lines.append("### Warnings")
            for w in warns:
                lines.append(f"- **{w.get('code')}**: {w.get('message')}" + (f" (`{w.get('path')}`)" if w.get("path") else ""))
            lines.append("")

    return "\n".join(lines)





def is_configuration_ok(reports: List[Dict[str, Any]]) -> bool:
    """
    Proste sprawdzenie czy wszystkie raporty są poprawne (brak błędów).
    """

    for r in reports:
        if not r.get("ok"):
            return False
    return True