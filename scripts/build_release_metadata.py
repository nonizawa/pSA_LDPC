#!/usr/bin/env python3
"""Build path-independent integrity metadata for the staged public release."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JSON_MANIFEST = ROOT / "release_manifest.json"
CHECKSUMS = ROOT / "CHECKSUMS.sha256"

ROLE_BY_PREFIX = {
    "configs/": "configuration",
    "data/metadata/": "instance and seed metadata",
    "data/processed/": "processed scientific record",
    "data/raw/": "compact raw scientific record",
    "docs/": "documentation and provenance",
    "figures/": "reference or regenerated figure",
    "scripts/": "reproduction and validation utility",
    "src/": "simulation or analysis source",
    "tests/": "lightweight validation test",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role(relative: str) -> str:
    for prefix, description in ROLE_BY_PREFIX.items():
        if relative.startswith(prefix):
            return description
    return "release metadata"


def package_files(exclude: set[Path]) -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path not in exclude
        and "__pycache__" not in path.parts
        and ".git" not in path.parts
    )


def main() -> None:
    size_audit = ROOT / 'PUBLIC_PACKAGE_SIZE_AUDIT.md'
    measured = package_files({JSON_MANIFEST, CHECKSUMS, size_audit})
    total = sum(path.stat().st_size for path in measured)
    lines = ['# Public package size audit', '',
             'Payload measurement excludes this size report and the two generated integrity manifests.', '',
             f'- Payload: {total:,} bytes ({total / 1048576:.3f} MiB)',
             f'- Payload files: {len(measured)}', '',
             '| Area | Bytes | MiB |', '|---|---:|---:|']
    for area in ['src', 'configs', 'data/processed', 'data/raw', 'figures', 'docs']:
        count = sum(p.stat().st_size for p in measured if p.is_relative_to(ROOT / area))
        lines.append(f'| {area} | {count:,} | {count / 1048576:.3f} |')
    lines += ['', '## Largest 20 files', '', '| File | Bytes |', '|---|---:|']
    for path in sorted(measured, key=lambda p: p.stat().st_size, reverse=True)[:20]:
        lines.append(f'| `{path.relative_to(ROOT)}` | {path.stat().st_size:,} |')
    for threshold in [10, 50, 100]:
        large = [p for p in measured if p.stat().st_size > threshold * 1000000]
        lines += ['', f'## Files exceeding {threshold} MB (decimal)', '', f'Count: {len(large)}']
        lines += [f'- `{p.relative_to(ROOT)}`: {p.stat().st_size:,} bytes' for p in large]
    lines += ['', '## Archive policy', '',
              'Large cycle-resolved source collections are excluded; see ZENODO_ARCHIVE_PLAN.md.',
              'All included files fit the requested standard-repository size ceiling. No Git LFS setup was performed.', '']
    size_audit.write_text('\n'.join(lines), encoding='utf-8')
    entries = []
    for path in package_files({JSON_MANIFEST, CHECKSUMS}):
        relative = path.relative_to(ROOT).as_posix()
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": role(relative),
            }
        )
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "package_label": "github_public_release_20260907",
        "manuscript_revision_label": "pra_manuscript_supplement_visual_cleanup_20260907",
        "file_count_excluding_manifests": len(entries),
        "total_bytes_excluding_manifests": sum(item["bytes"] for item in entries),
        "files": entries,
    }
    JSON_MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    checksum_paths = package_files({CHECKSUMS})
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in checksum_paths]
    CHECKSUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {JSON_MANIFEST.name} and {CHECKSUMS.name} for {len(checksum_paths)} files.")


if __name__ == "__main__":
    main()
