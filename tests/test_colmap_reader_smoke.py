import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def camera_signature(cameras):
    rows = []
    for cam in sorted(cameras, key=lambda item: item.image_name):
        rows.append({
            "image_name": cam.image_name,
            "uid": int(cam.uid),
            "width": int(cam.width),
            "height": int(cam.height),
            "fovx": float(cam.FovX),
            "fovy": float(cam.FovY),
            "r": np.asarray(cam.R, dtype=np.float64).round(10).tolist(),
            "t": np.asarray(cam.T, dtype=np.float64).round(10).tolist(),
        })
    return rows


def point_count(point_cloud):
    if point_cloud is None:
        return None
    return int(np.asarray(point_cloud.points).shape[0])


def current_reader_summary(scene_path, images, sparse_subdir):
    from scene.dataset_readers import readColmapSceneInfo, resolve_colmap_sparse_model_path

    info = readColmapSceneInfo(
        str(scene_path),
        images,
        eval=False,
        sparse_subdir=sparse_subdir,
    )
    sparse_path = resolve_colmap_sparse_model_path(str(scene_path), sparse_subdir)
    cameras = info.train_cameras + info.test_cameras
    return {
        "reader": "Improved_GS-W",
        "sparse_path": sparse_path,
        "camera_count": len(cameras),
        "point_count": point_count(info.point_cloud),
        "image_names": sorted(cam.image_name for cam in cameras),
        "camera_signature": camera_signature(cameras),
    }


def official_reader_summary(scene_path, official_repo):
    code = r"""
import json
import numpy as np
import sys
from scene.dataset_readers import readColmapSceneInfo

def camera_signature(cameras):
    rows = []
    for cam in sorted(cameras, key=lambda item: item.image_name):
        rows.append({
            "image_name": cam.image_name,
            "uid": int(cam.uid),
            "width": int(cam.width),
            "height": int(cam.height),
            "fovx": float(cam.FovX),
            "fovy": float(cam.FovY),
            "r": np.asarray(cam.R, dtype=np.float64).round(10).tolist(),
            "t": np.asarray(cam.T, dtype=np.float64).round(10).tolist(),
        })
    return rows

def point_count(point_cloud):
    if point_cloud is None:
        return None
    return int(np.asarray(point_cloud.points).shape[0])

scene_path = sys.argv[1]
info = readColmapSceneInfo(scene_path, "images", "", False, False)
cameras = info.train_cameras + info.test_cameras
summary = {
    "reader": "official_3dgs",
    "camera_count": len(cameras),
    "point_count": point_count(info.point_cloud),
    "image_names": sorted(cam.image_name for cam in cameras),
    "camera_signature": camera_signature(cameras),
}
print("JSON_SUMMARY_START" + json.dumps(summary, sort_keys=True))
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code, str(scene_path)],
        cwd=str(official_repo),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "official 3DGS reader failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("JSON_SUMMARY_START"):
            return json.loads(line[len("JSON_SUMMARY_START"):])
    raise RuntimeError(f"official 3DGS reader did not emit JSON summary:\n{result.stdout}")


def compare_summaries(current, official):
    errors = []
    for key in ("camera_count", "point_count", "image_names"):
        if current[key] != official[key]:
            errors.append(f"{key} mismatch")

    if len(current["camera_signature"]) == len(official["camera_signature"]):
        for cur, ref in zip(current["camera_signature"], official["camera_signature"]):
            for key in ("image_name", "uid", "width", "height"):
                if cur[key] != ref[key]:
                    errors.append(f"camera {cur.get('image_name')} {key} mismatch: {cur[key]} != {ref[key]}")
            for key in ("fovx", "fovy"):
                if abs(cur[key] - ref[key]) > 1e-9:
                    errors.append(f"camera {cur.get('image_name')} {key} mismatch: {cur[key]} != {ref[key]}")
            for key in ("r", "t"):
                if not np.allclose(np.asarray(cur[key]), np.asarray(ref[key]), atol=1e-9):
                    errors.append(f"camera {cur.get('image_name')} {key} mismatch")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--official-repo", default=Path(r"G:\wl3dgs\3dgs_original"), type=Path)
    parser.add_argument("--images", default="images")
    parser.add_argument("--sparse-subdir", default="")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    current = current_reader_summary(args.scene, args.images, args.sparse_subdir)
    official = official_reader_summary(args.scene, args.official_repo)
    errors = compare_summaries(current, official)
    report = {
        "scene": str(args.scene),
        "current": current,
        "official": official,
        "errors": errors,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
