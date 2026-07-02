from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
WL3DGS = REPO.parent
REPORT = REPO / "reports" / "baseline_completion"
SPLIT_ROOT = WL3DGS / "splits" / "max1600_llffhold8_v1"
REGISTRY = REPORT / "GSW_12SCENE_RUN_REGISTRY.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalized_names(names: list[Any]) -> list[str]:
    return sorted(str(name).replace("\\", "/").split("/")[-1] for name in names)


def canonical_payload(scene_name: str, train_names: list[str], test_names: list[str]) -> bytes:
    payload = {
        "scene_name": scene_name,
        "test_images": normalized_names(test_names),
        "train_images": normalized_names(train_names),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_hash(scene_name: str, train_names: list[str], test_names: list[str]) -> tuple[str, str]:
    payload = canonical_payload(scene_name, train_names, test_names)
    return sha256_bytes(payload), payload.decode("utf-8")


def line_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_manifest(path: Path) -> tuple[str, list[str], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    scene = data.get("scene") or data.get("scene_name") or data.get("scene_id") or path.stem.replace("_SPLIT", "")
    train = data.get("train_images") or data.get("train_names") or []
    test = data.get("test_images") or data.get("test_names") or []
    return str(scene), [str(x) for x in train], [str(x) for x in test]


def row_from_split_text(scene_name: str) -> dict[str, Any]:
    split_dir = SPLIT_ROOT / scene_name
    train = line_names(split_dir / "train.txt")
    test = line_names(split_dir / "test.txt")
    canonical, payload = canonical_hash(scene_name, train, test)
    return {
        "scene_name": scene_name,
        "source_id": "frozen_split_text",
        "source_path": str(split_dir),
        "raw_manifest_sha256": "",
        "canonical_split_sha256": canonical,
        "train_count": len(train),
        "test_count": len(test),
        "train_images": ";".join(normalized_names(train)),
        "test_images": ";".join(normalized_names(test)),
        "canonical_payload_json": payload,
        "status": "ok" if train and test else "missing_train_or_test",
    }


def row_from_manifest(scene_name_hint: str, source_id: str, path: Path) -> dict[str, Any]:
    scene, train, test = load_manifest(path)
    scene_name = scene_name_hint or scene
    canonical, payload = canonical_hash(scene_name, train, test)
    return {
        "scene_name": scene_name,
        "source_id": source_id,
        "source_path": str(path),
        "raw_manifest_sha256": sha256_file(path),
        "canonical_split_sha256": canonical,
        "train_count": len(train),
        "test_count": len(test),
        "train_images": ";".join(normalized_names(train)),
        "test_images": ";".join(normalized_names(test)),
        "canonical_payload_json": payload,
        "status": "ok" if train and test else "missing_train_or_test",
    }


def special_trackmobile_rows() -> list[dict[str, Any]]:
    scene = "self_Trackmobile_4650TM_Mobile_Railcar_Mover"
    candidates = [
        ("legacy_repo_trackmobile_split", REPO / "splits" / "TRACKMOBILE_SPLIT.json"),
        ("legacy_review_trackmobile_split_test", WL3DGS / "gpt_review_packages" / "trackmobile_split_test.json"),
    ]
    rows: list[dict[str, Any]] = []
    for source_id, path in candidates:
        if not path.exists():
            continue
        try:
            rows.append(row_from_manifest(scene, source_id, path))
        except Exception as exc:
            rows.append(
                {
                    "scene_name": scene,
                    "source_id": source_id,
                    "source_path": str(path),
                    "raw_manifest_sha256": sha256_file(path),
                    "canonical_split_sha256": "",
                    "train_count": 0,
                    "test_count": 0,
                    "train_images": "",
                    "test_images": "",
                    "canonical_payload_json": "",
                    "status": f"error:{exc}",
                }
            )
    return rows


def generate(report_dir: Path = REPORT, registry_path: Path = REGISTRY) -> list[dict[str, Any]]:
    registry = read_csv(registry_path)
    scene_names = [row["scene_name"] for row in registry]
    rows: list[dict[str, Any]] = []
    for scene in scene_names:
        rows.append(row_from_split_text(scene))
        manifest = next((Path(row["manifest_path"]) for row in registry if row["scene_name"] == scene), None)
        if manifest and manifest.exists():
            rows.append(row_from_manifest(scene, "registry_manifest", manifest))
    rows.extend(special_trackmobile_rows())
    write_csv(report_dir / "CANONICAL_SPLIT_HASHES.csv", rows)

    scene_status: list[str] = []
    for scene in scene_names:
        scene_rows = [row for row in rows if row["scene_name"] == scene and row["status"] == "ok"]
        unique = sorted({row["canonical_split_sha256"] for row in scene_rows})
        scene_status.append(
            f"| {scene} | {len(scene_rows)} | {len(unique)} | {'PASS' if len(unique) == 1 else 'FAIL'} | "
            f"{'<br>'.join(unique)} |"
        )

    track_rows = [row for row in rows if row["scene_name"] == "self_Trackmobile_4650TM_Mobile_Railcar_Mover"]
    track_unique = sorted({row["canonical_split_sha256"] for row in track_rows if row["status"] == "ok"})
    raw_pairs = [
        f"`{row['source_id']}` raw=`{row['raw_manifest_sha256'] or 'n/a'}` canonical=`{row['canonical_split_sha256']}`"
        for row in track_rows
    ]
    track_pass = len(track_unique) == 1
    md = [
        "# CANONICAL_SPLIT_HASH_AUDIT",
        "",
        "Canonical split hash is computed from only `scene_name`, lexicographically sorted `train_images`, and lexicographically sorted `test_images`.",
        "Serialization is UTF-8 JSON with `sort_keys=True` and `separators=(',', ':')`; no absolute paths, timestamps, metadata, or formatting-sensitive fields are included.",
        "",
        f"- Trackmobile canonical equivalence gate: `{'PASS' if track_pass else 'FAIL'}`",
        f"- Trackmobile unique canonical hashes: `{len(track_unique)}`",
        "",
        "## Trackmobile Raw vs Canonical",
        "",
    ]
    md.extend([f"- {item}" for item in raw_pairs])
    md.extend(
        [
            "",
            "## Scene Summary",
            "",
            "| scene | sources | unique canonical hashes | status | canonical hash |",
            "|---|---:|---:|---|---|",
            *scene_status,
            "",
        ]
    )
    write_text(report_dir / "CANONICAL_SPLIT_HASH_AUDIT.md", "\n".join(md))
    return rows


def main() -> int:
    rows = generate()
    track = [row for row in rows if row["scene_name"] == "self_Trackmobile_4650TM_Mobile_Railcar_Mover" and row["status"] == "ok"]
    if len({row["canonical_split_sha256"] for row in track}) != 1:
        raise SystemExit("Trackmobile canonical split mismatch")
    print(REPORT / "CANONICAL_SPLIT_HASH_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
