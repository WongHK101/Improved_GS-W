from pathlib import Path
import sys
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "audit"))

import gsw_equivalence_audit as audit  # noqa: E402


def test_read_cfg_namespace_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "cfg_args"
        cfg.write_text("Namespace(use_colors_precomp=True, map_num=3, nested={'a': 1})", encoding="utf-8")
        parsed = audit.read_cfg(cfg)
    assert parsed["use_colors_precomp"] is True
    assert parsed["map_num"] == 3
    assert parsed["nested"] == {"a": 1}


def test_read_tsv_split_filters_null_id():
    with tempfile.TemporaryDirectory() as tmp:
        tsv = Path(tmp) / "scene.tsv"
        tsv.write_text(
            "filename\tid\tsplit\n"
            "0001.jpg\t1\ttest\n"
            "bad.jpg\t\ttrain\n"
            "0002.jpg\t2\ttrain\n",
            encoding="utf-8",
        )
        frame = audit.read_tsv_split(tsv)
    assert frame["filename"].tolist() == ["0001.jpg", "0002.jpg"]
    assert frame["id"].tolist() == [1, 2]
    assert frame["split"].tolist() == ["test", "train"]


def test_metric_images_identical_pair():
    with tempfile.TemporaryDirectory() as tmp:
        render = Path(tmp) / "render.png"
        gt = Path(tmp) / "gt.png"
        img = Image.new("RGB", (4, 3), (10, 20, 30))
        img.save(render)
        img.save(gt)
        metrics = audit.metric_images(render, gt)
    assert metrics["mse"] == 0.0
    assert metrics["psnr"] == float("inf")
    assert metrics["mae"] == 0.0
