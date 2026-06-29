# Active Branch Audit

This audit checks the existing clean 30k code path without modifying training or running a new 30k job.

## Branch evidence

| Feature | Location | Evidence | Risk |
|---|---:|---|---|
| map_generator init | `scene\gaussian_model.py:84` | `self.map_generator=Unet_model` | medium |
| map_generator optimizer | `scene\gaussian_model.py:260` | `"name": "map_generator"` | medium |
| map_generator forward | `scene\gaussian_model.py:456` | `out_gen=self.map_generator` | high if accidentally disabled |
| feature mask | `scene\gaussian_model.py:460` | `self.features_mask=out_gen["mask"]` | medium |
| kmap/pjmap sampling | `scene\gaussian_model.py:482` | `project2d(` | medium |
| color_net init | `scene\gaussian_model.py:108` | `self.color_net=Color_net` | medium |
| color_net forward | `scene\gaussian_model.py:492` | `self._pre_comp_color=self.color_net` | high if accidentally disabled |
| colors_precomp renderer | `gaussian_renderer\__init__.py:111` | `pc.use_colors_precomp` | high if accidentally disabled |
| LPIPS train loss | `train.py:165` | `lpips_criteria(image,gt_image)` | medium |
| box-coordinate loss | `train.py:167` | `use_box_coord_loss` | medium |
| densification | `scene\gaussian_model.py:740` | `def densify_and_prune` | medium |
| checkpoint save map | `scene\gaussian_model.py:368` | `map_generator.state_dict` | medium |
| checkpoint load map | `scene\gaussian_model.py:430` | `map_generator.load_state_dict` | medium |
| dropout eval disable | `scene\gaussian_model.py:773` | `self.color_net.use_drop_out=False` | medium |

## Checkpoint branch artifacts

- Historical map generator exists: `True`, size `63263081` bytes.
- Current map generator exists: `True`, size `63263081` bytes.
- Historical color net exists: `True`, size `222807` bytes.
- Current color net exists: `True`, size `222807` bytes.
- Historical `box_coord` shape: `(133906, 3, 2)`.
- Current `box_coord` shape: `(140945, 3, 2)`.

## Direct gradient diagnostic status

No new training or optimizer step was executed in this audit. The script did not run a backward pass on the 30k checkpoint. The branch is nevertheless active by source-path inspection: `map_generator(img)` feeds `_point_features`, `_point_features` feeds `Color_net`, and the rendered image participates in the photometric and LPIPS losses. Existing checkpoint files also contain map-generator, color-net and box-coordinate states.

## Evaluation-mode behavior

`GaussianModel.set_eval(True)` switches `color_net` to eval, disables color-net dropout, and disables the feature mask branch. It is restored by `set_eval(False)`. Clean strict modes additionally prevent target test RGB from entering `map_generator`.
