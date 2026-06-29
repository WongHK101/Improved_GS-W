#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image,ImageDraw
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import pandas as pd
import glob
import hashlib

COLMAP_BINARY_FILES = ("images.bin", "cameras.bin")
COLMAP_TEXT_FILES = ("images.txt", "cameras.txt")

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    split_manifest_path: str = ""
    split_summary: dict = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)     
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def has_colmap_sparse_model(sparse_path):
    return (
        all(os.path.exists(os.path.join(sparse_path, name)) for name in COLMAP_BINARY_FILES)
        or all(os.path.exists(os.path.join(sparse_path, name)) for name in COLMAP_TEXT_FILES)
    )

def resolve_colmap_sparse_model_path(path, sparse_subdir=""):
    candidates = []
    if sparse_subdir:
        explicit_path = sparse_subdir if os.path.isabs(sparse_subdir) else os.path.join(path, sparse_subdir)
        candidates.append(("explicit", explicit_path))
    candidates.extend([
        ("sparse/0", os.path.join(path, "sparse", "0")),
        ("sparse", os.path.join(path, "sparse")),
    ])

    available = [(label, sparse_path) for label, sparse_path in candidates if has_colmap_sparse_model(sparse_path)]
    if sparse_subdir and not available:
        raise FileNotFoundError(f"Explicit COLMAP sparse model path is invalid or incomplete: {explicit_path}")
    if not available:
        checked = ", ".join(sparse_path for _, sparse_path in candidates)
        raise FileNotFoundError(f"No COLMAP sparse model found. Checked: {checked}")

    selected_label, selected_path = available[0]
    if len(available) > 1:
        print("[COLMAP] Multiple sparse model candidates found: " + ", ".join(f"{label}={sparse_path}" for label, sparse_path in available))
    print(f"[COLMAP] Using sparse model path ({selected_label}): {selected_path}")
    return selected_path

def normalize_image_name(name):
    return str(name).replace("\\", "/").split("/")[-1]

def ordered_unique_or_error(names, label):
    seen = set()
    ordered = []
    duplicates = []
    for name in names:
        normalized = normalize_image_name(name)
        if normalized in seen:
            duplicates.append(normalized)
        seen.add(normalized)
        ordered.append(normalized)
    if duplicates:
        raise ValueError(f"Duplicate image names in {label}: {sorted(set(duplicates))}")
    return ordered

def sha256_lines(lines):
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def load_frozen_split_manifest(split_file):
    if not split_file:
        raise ValueError("--split_file is required when --split_mode frozen_manifest is used.")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split manifest does not exist: {split_file}")
    with open(split_file, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    train_names = ordered_unique_or_error(manifest.get("train_images", []), "train_images")
    test_names = ordered_unique_or_error(manifest.get("test_images", []), "test_images")
    if not train_names:
        raise ValueError("Frozen split manifest has an empty train_images list.")
    if not test_names:
        raise ValueError("Frozen split manifest has an empty test_images list.")
    overlap = set(train_names).intersection(test_names)
    if overlap:
        raise ValueError(f"Frozen split manifest train/test overlap: {sorted(overlap)}")
    return manifest, train_names, test_names

def split_cameras_from_manifest(cam_infos, split_file):
    manifest, train_names, test_names = load_frozen_split_manifest(split_file)
    camera_by_name = {}
    duplicates = []
    for cam in cam_infos:
        name = normalize_image_name(cam.image_name)
        if name in camera_by_name:
            duplicates.append(name)
        camera_by_name[name] = cam
    if duplicates:
        raise ValueError(f"Duplicate registered camera image names: {sorted(set(duplicates))}")

    registered = set(camera_by_name)
    manifest_names = set(train_names).union(test_names)
    missing = sorted(registered - manifest_names)
    extra = sorted(manifest_names - registered)
    if missing:
        raise ValueError(f"Frozen split manifest is missing registered images: {missing}")
    if extra:
        raise ValueError(f"Frozen split manifest contains images not registered by COLMAP: {extra}")

    train_cameras = [camera_by_name[name] for name in train_names]
    test_cameras = [camera_by_name[name] for name in test_names]
    summary = {
        "split_mode": "frozen_manifest",
        "manifest_path": os.path.abspath(split_file),
        "manifest_sha256": hashlib.sha256(Path(split_file).read_bytes()).hexdigest(),
        "registered_count": len(cam_infos),
        "train_count": len(train_cameras),
        "test_count": len(test_cameras),
        "registered_sha256": sha256_lines(sorted(registered)),
        "train_sha256": sha256_lines(train_names),
        "test_sha256": sha256_lines(test_names),
        "train_images": train_names,
        "test_images": test_names,
        "protocol": manifest.get("protocol", ""),
        "ordering_rule": manifest.get("ordering_rule", ""),
    }
    return train_cameras, test_cameras, manifest, summary

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
  
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, extr.name)
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = extr.name
        image = Image.open(image_path)        

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)     
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, sparse_subdir="", split_mode="legacy", split_file=""):
    sparse_path = resolve_colmap_sparse_model_path(path, sparse_subdir)
    try:
        cameras_extrinsic_file = os.path.join(sparse_path, "images.bin")
        cameras_intrinsic_file = os.path.join(sparse_path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)     
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)    
    except:
        cameras_extrinsic_file = os.path.join(sparse_path, "images.txt")
        cameras_intrinsic_file = os.path.join(sparse_path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    split_summary = {"split_mode": split_mode}
    if split_mode == "frozen_manifest":
        train_cam_infos, test_cam_infos, _, split_summary = split_cameras_from_manifest(cam_infos, split_file)
        print(
            "[Split] frozen_manifest loaded: "
            f"train={len(train_cam_infos)} test={len(test_cam_infos)} "
            f"manifest={split_file}"
        )
    elif split_mode == "legacy" and eval:
        root_dir=os.path.dirname(path)
        tsv = glob.glob(os.path.join(root_dir, '*.tsv'))[0]
        scene_name = os.path.basename(tsv)[:-4]  
        files = pd.read_csv(tsv, sep='\t')                          
        files = files[~files['id'].isnull()]   
        files.reset_index(inplace=True, drop=True)

        img_path_to_id = {}
        for v in cam_extrinsics.values():
            img_path_to_id[v.name] = v.id
        img_ids = []
        image_paths = {} # {id: filename}
        for filename in list(files['filename']):
            if filename in img_path_to_id:
                id_ = img_path_to_id[filename]
                image_paths[id_] = filename              
                img_ids += [id_]               

        img_ids_train = [id_ for i, id_ in enumerate(img_ids) 
                                    if files.loc[i, 'split']=='train']
        img_ids_test = [id_ for i, id_ in enumerate(img_ids)
                                    if files.loc[i, 'split']=='test']
        
        train_cam_infos =[ c for c in cam_infos if c.uid in img_ids_train]
        test_cam_infos =[ c for c in cam_infos if c.uid in img_ids_test]
        split_summary = {
            "split_mode": "legacy_tsv",
            "train_count": len(train_cam_infos),
            "test_count": len(test_cam_infos),
        }
    elif split_mode == "legacy":
        train_cam_infos = cam_infos
        test_cam_infos = []
        split_summary = {
            "split_mode": "legacy_all_train",
            "train_count": len(train_cam_infos),
            "test_count": len(test_cam_infos),
        }
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(sparse_path, "points3D.ply")
    bin_path = os.path.join(sparse_path, "points3D.bin")
    txt_path = os.path.join(sparse_path, "points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           split_manifest_path=os.path.abspath(split_file) if split_file else "",
                           split_summary=split_summary)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png",data_perturb=None,split="train"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]      
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)     

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])         
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            if idx != 0 and split=="train":
                image=add_perturbation(image,data_perturb,idx)
                   
            im_data = np.array(image.convert("RGBA"))  
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])      
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png",data_perturb=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension,data_perturb=data_perturb,split="train")  #[CameraInfo(id，fov，R，T，图片路径，图片，高，宽)...100长度]
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension,data_perturb=None,split="test")
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)          

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3         
        shs = np.random.random((num_pts, 3)) / 255.0              
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)            
    try:
        pcd = fetchPly(ply_path)                
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
def add_perturbation(img, perturbation, seed):
    if 'occ' in perturbation:
        draw = ImageDraw.Draw(img)
        np.random.seed(seed)
        left = np.random.randint(200, 400)
        top = np.random.randint(200, 400)
        for i in range(10):
            np.random.seed(10*seed+i)
            random_color = tuple(np.random.choice(range(256), 3))
            draw.rectangle(((left+20*i, top), (left+20*(i+1), top+200)),
                            fill=random_color)

    if 'color' in perturbation:
        np.random.seed(seed)
        img_np = np.array(img)/255.0     #H,W,4
        s = np.random.uniform(0.8, 1.2, size=3)   #
        b = np.random.uniform(-0.2, 0.2, size=3)   #
        img_np[..., :3] = np.clip(s*img_np[..., :3]+b, 0, 1)
        img = Image.fromarray((255*img_np).astype(np.uint8))

    return img
