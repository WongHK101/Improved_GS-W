import argparse
import inspect
import json
import math
from argparse import ArgumentParser
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def tensor_summary(tensor):
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "finite": bool(torch.isfinite(tensor).all().item()),
    }


def grad_summary(name, tensor):
    grad = tensor.grad
    return {
        "name": name,
        "present": grad is not None,
        "shape": list(grad.shape) if grad is not None else None,
        "finite": bool(torch.isfinite(grad).all().item()) if grad is not None else False,
        "abs_max": float(grad.detach().abs().max().item()) if grad is not None else None,
    }


def main():
    from arguments import ModelParams, PipelineParams
    from arguments import args_init

    parser = ArgumentParser(description="Rasterizer API compatibility regression test")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--data_perturb", nargs="+", default=[])
    args = parser.parse_args()
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    args = args_init.argument_init(args)

    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from gaussian_renderer import raster_settings_kwargs, rasterizer_color_and_radii
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, args)
    scene = Scene(dataset, gaussians, shuffle=False)
    view = scene.getTrainCameras()[0]

    gaussians.forward(view)
    screenspace_points = torch.zeros_like(
        gaussians.get_xyz,
        dtype=gaussians.get_xyz.dtype,
        requires_grad=True,
        device="cuda",
    )
    screenspace_points.retain_grad()

    raster_settings = GaussianRasterizationSettings(**raster_settings_kwargs(
        image_height=int(view.image_height),
        image_width=int(view.image_width),
        tanfovx=math.tan(view.FoVx * 0.5),
        tanfovy=math.tan(view.FoVy * 0.5),
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=view.world_view_transform,
        projmatrix=view.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=view.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    ))

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    outputs = rasterizer(
        means3D=gaussians.get_xyz_dealed,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=gaussians.get_colors,
        opacities=gaussians.get_opacity_dealed,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        cov3D_precomp=None,
    )
    color, radii = rasterizer_color_and_radii(outputs)
    invdepths = outputs[2] if len(outputs) > 2 else None

    loss = color.mean()
    loss.backward()

    summaries = {
        "rasterizer_module": str(sys.modules[GaussianRasterizer.__module__].__file__),
        "settings_fields": list(getattr(GaussianRasterizationSettings, "_fields", ())),
        "forward_signature": str(inspect.signature(GaussianRasterizer.forward)),
        "return_tuple_len": len(outputs),
        "return_semantics": ["color", "radii"] + (["invdepths"] if len(outputs) > 2 else []),
        "color": tensor_summary(color),
        "radii": tensor_summary(radii),
        "invdepths": tensor_summary(invdepths),
        "loss": float(loss.item()),
        "grads": [
            grad_summary("xyz", gaussians._xyz),
            grad_summary("opacity", gaussians._opacity),
            grad_summary("scaling", gaussians._scaling),
            grad_summary("rotation", gaussians._rotation),
            grad_summary("features_intrinsic", gaussians._features_intrinsic),
        ],
        "screenspace_grad": tensor_summary(screenspace_points.grad),
    }

    errors = []
    if summaries["return_tuple_len"] < 2:
        errors.append("Rasterizer returned fewer than two outputs.")
    if color.shape != (3, view.image_height, view.image_width):
        errors.append(f"Unexpected color shape: {tuple(color.shape)}")
    if radii.shape[0] != gaussians.get_xyz.shape[0]:
        errors.append(f"Unexpected radii shape: {tuple(radii.shape)}")
    for key in ("color", "radii"):
        if not summaries[key]["finite"]:
            errors.append(f"{key} contains non-finite values.")
    if invdepths is not None and not summaries["invdepths"]["finite"]:
        errors.append("invdepths contains non-finite values.")
    for item in summaries["grads"]:
        if not item["present"]:
            errors.append(f"{item['name']} grad is missing.")
        elif not item["finite"]:
            errors.append(f"{item['name']} grad is non-finite.")
    if not summaries["screenspace_grad"]["finite"]:
        errors.append("screenspace gradient is non-finite.")

    summaries["errors"] = errors
    text = json.dumps(summaries, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
