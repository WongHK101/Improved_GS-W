import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def write_manifest(path, train_images, test_images):
    path.write_text(json.dumps({
        "scene": "unit_test",
        "protocol": "llff_hold_8",
        "ordering_rule": "unit test",
        "train_images": train_images,
        "test_images": test_images,
    }, indent=2), encoding="utf-8")


def expect_error(label, func, contains):
    try:
        func()
    except Exception as exc:
        if contains not in str(exc):
            raise AssertionError(f"{label}: expected error containing {contains!r}, got {exc!r}")
        return str(exc)
    raise AssertionError(f"{label}: expected an error")


def assert_names(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label} mismatch: {actual} != {expected}")


def main():
    parser = argparse.ArgumentParser(description="Frozen split manifest tests")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    from scene.dataset_readers import readColmapSceneInfo, split_cameras_from_manifest

    info = readColmapSceneInfo(
        str(args.scene),
        "images",
        eval=False,
        sparse_subdir="",
        split_mode="frozen_manifest",
        split_file=str(args.manifest),
    )
    train_names = [cam.image_name for cam in info.train_cameras]
    test_names = [cam.image_name for cam in info.test_cameras]

    expected_train = [
        "0002.jpg",
        "0003.jpg",
        "0004.jpg",
        "0005.jpg",
        "0006.jpg",
        "0007.jpg",
        "0008.jpg",
        "0010.jpg",
        "0011.jpg",
        "0012.jpg",
        "0013.jpg",
        "0014.jpg",
        "0015.jpg",
    ]
    expected_test = ["0001.jpg", "0009.jpg"]
    assert_names(train_names, expected_train, "train_names")
    assert_names(test_names, expected_test, "test_names")
    if set(train_names).intersection(test_names):
        raise AssertionError("train/test overlap detected")

    shuffled = list(reversed(info.train_cameras + info.test_cameras))
    train_shuffled, test_shuffled, _, _ = split_cameras_from_manifest(shuffled, str(args.manifest))
    assert_names([cam.image_name for cam in train_shuffled], expected_train, "shuffled train_names")
    assert_names([cam.image_name for cam in test_shuffled], expected_test, "shuffled test_names")

    temp_dir = args.output_json.parent / "split_manifest_negative_cases" if args.output_json else args.manifest.parent / "split_manifest_negative_cases"
    temp_dir.mkdir(parents=True, exist_ok=True)

    negative_results = {}
    overlap_path = temp_dir / "overlap.json"
    write_manifest(overlap_path, ["0001.jpg", "0002.jpg"], ["0001.jpg"])
    negative_results["overlap"] = expect_error(
        "overlap",
        lambda: readColmapSceneInfo(str(args.scene), "images", False, split_mode="frozen_manifest", split_file=str(overlap_path)),
        "overlap",
    )

    duplicate_path = temp_dir / "duplicate.json"
    write_manifest(duplicate_path, ["0001.jpg", "0001.jpg"], ["0002.jpg"])
    negative_results["duplicate"] = expect_error(
        "duplicate",
        lambda: readColmapSceneInfo(str(args.scene), "images", False, split_mode="frozen_manifest", split_file=str(duplicate_path)),
        "Duplicate image names",
    )

    missing_path = temp_dir / "missing.json"
    write_manifest(missing_path, expected_train, ["0001.jpg"])
    negative_results["missing_registered"] = expect_error(
        "missing_registered",
        lambda: readColmapSceneInfo(str(args.scene), "images", False, split_mode="frozen_manifest", split_file=str(missing_path)),
        "missing registered images",
    )

    extra_path = temp_dir / "extra.json"
    write_manifest(extra_path, expected_train, expected_test + ["9999.jpg"])
    negative_results["extra"] = expect_error(
        "extra",
        lambda: readColmapSceneInfo(str(args.scene), "images", False, split_mode="frozen_manifest", split_file=str(extra_path)),
        "not registered",
    )

    empty_train_path = temp_dir / "empty_train.json"
    write_manifest(empty_train_path, [], expected_test)
    negative_results["empty_train"] = expect_error(
        "empty_train",
        lambda: readColmapSceneInfo(str(args.scene), "images", False, split_mode="frozen_manifest", split_file=str(empty_train_path)),
        "empty train_images",
    )

    empty_test_path = temp_dir / "empty_test.json"
    write_manifest(empty_test_path, expected_train, [])
    negative_results["empty_test"] = expect_error(
        "empty_test",
        lambda: readColmapSceneInfo(str(args.scene), "images", False, split_mode="frozen_manifest", split_file=str(empty_test_path)),
        "empty test_images",
    )

    windows_path = temp_dir / "windows_paths.json"
    write_manifest(
        windows_path,
        [f"C:\\tmp\\{name}" for name in expected_train],
        [f"C:\\tmp\\{name}" for name in expected_test],
    )
    win_info = readColmapSceneInfo(
        str(args.scene),
        "images",
        False,
        split_mode="frozen_manifest",
        split_file=str(windows_path),
    )
    assert_names([cam.image_name for cam in win_info.train_cameras], expected_train, "windows path train_names")
    assert_names([cam.image_name for cam in win_info.test_cameras], expected_test, "windows path test_names")

    report = {
        "scene": str(args.scene),
        "manifest": str(args.manifest),
        "manifest_sha256": info.split_summary["manifest_sha256"],
        "train_count": len(train_names),
        "test_count": len(test_names),
        "train_names": train_names,
        "test_names": test_names,
        "negative_cases": negative_results,
        "windows_path_matching": "passed",
        "reader_order_invariance": "passed",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
