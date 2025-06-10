# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data parser for TruckScenes dataset"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Set, Tuple, Type

import numpy as np
import pypcd4
import pyquaternion
import torch
from truckscenes.truckscenes import TruckScenes as TruckScenesDatabase

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType, transform_points
from nerfstudio.data.dataparsers.ad_dataparser import (
    DUMMY_DISTANCE_VALUE,
    OPENCV_TO_NERFSTUDIO,
    ADDataParser,
    ADDataParserConfig,
    SplitTypes,
)
from nerfstudio.data.utils.lidar_elevation_mappings import OUSTER_OS0_ELEVATION_MAPPING, PANDAR64_ELEVATION_MAPPING
from nerfstudio.utils import poses as pose_utils

ALLOWED_RIGID_CLASSES = (
    "vehicle.car",
    "vehicle.bicycle",
    "vehicle.motorcycle",
    "vehicle.bus",
    "vehicle.truck",
    "vehicle.train",
    "vehicle.ego_trailer",
    "vehicle.trailer",
    "vehicle.construction",
    "vehicle.othervehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "movable_object.pushable_pullable",
)
ALLOWED_DEFORMABLE_CLASSES = ("human.pedestrian",)

TRACKING_TO_GT_CLASSNAME_MAPPING = {
    "pedestrian": "human.pedestrian",
    "bicycle": "vehicle.bicycle",
    "motorcycle": "vehicle.motorcycle",
    "car": "vehicle.car",
    "bus": "vehicle.bus",
    "truck": "vehicle.truck",
    "trailer": "vehicle.truck",
    "construction_vehicle": "vehicle.construction",
    "emergency_vehicle": "vehicle.othervehicle.emergency.ambulance",
    "police_vehicle": "vehicle.emergency.police",
    "other_vehicle": "vehicle.othervehicle",
    "ego_trailer": "vehicle.ego_trailer",
    "train": "vehicle.train",
    "pushable_pullable": "movable_object.pushable_pullable",
}
# Nuscenes defines actor coordinate system as x-forward, y-left, z-up
# But we want to use x-right, y-forward, z-up
# So we need to rotate the actor coordinate system by 90 degrees around z-axis
WLH_TO_LWH = np.array(
    [
        [0, 1.0, 0, 0],
        [-1.0, 0, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.0],
    ]
)
HORIZONTAL_BEAM_DIVERGENCE = 0.00333333333  # radians, given as 4 inches at 100 feet
VERTICAL_BEAM_DIVERGENCE = 0.00166666666  # radians, given as 2 inches at 100 feet

TRUCKSCENES_ELEVATION_MAPPING = {
    "LEFT": PANDAR64_ELEVATION_MAPPING, # Field of View: 360° x 40° | Resolution: 64 vertical layers
    "RIGHT": PANDAR64_ELEVATION_MAPPING,
    "REAR": OUSTER_OS0_ELEVATION_MAPPING, # Field of View: 360° x 90° | Resolution: 64 vertical layers | Range: 35 m @10 %
    "TOP_FRONT": OUSTER_OS0_ELEVATION_MAPPING, 
    "TOP_LEFT": OUSTER_OS0_ELEVATION_MAPPING,
    "TOP_RIGHT": OUSTER_OS0_ELEVATION_MAPPING,
}
TRUCKSCENES_AZIMUTH_RESOLUTION = {
    "LEFT": 1 / 3.0,
    "RIGHT": 1 / 3.0,
    "REAR": 1 / 3.0,  # TODO: check these values
    "TOP_FRONT": 1 / 3.0,  # TODO: check these values
    "TOP_LEFT": 1 / 3.0,  # TODO: check these values
    "TOP_RIGHT": 1 / 3.0,  # TODO: check these values
}
TRUCKSCENES_SKIP_ELEVATION_CHANNELS = {k: tuple() for k in TRUCKSCENES_ELEVATION_MAPPING.keys()}


# all are 1980×943 undistorted
AVAILABLE_CAMERAS = (
    "LEFT_FRONT",
    "RIGHT_FRONT",
    "LEFT_BACK",
    "RIGHT_BACK",
)
# see ELEVATION_MAPPING for the elevation mapping of each lidar
AVAILABLE_LIDARS = (
    "TOP_FRONT",
    "TOP_LEFT",
    "TOP_RIGHT",
    "LEFT",
    "RIGHT",
    "REAR",
) 

AVAILABLE_RADARS = (
    "LEFT_FRONT",
    "LEFT_BACK",
    "LEFT_SIDE",
    "RIGHT_FRONT",
    "RIGHT_BACK",
    "RIGHT_SIDE",
)

CAMERA_TO_BOTTOM_RIGHT_CROP = {k: (0, 0) for k in AVAILABLE_CAMERAS}
CAMERA_TO_BOTTOM_RIGHT_CROP["RIGHT_BACK"] = (0, 50) # usually catch some parts of the trailer.

DEFAULT_IMAGE_HEIGHT = 943
DEFAULT_IMAGE_WIDTH = 1980

@dataclass
class TruckScenesDataParserConfig(ADDataParserConfig):
    """TruckScenes dataset config.

    more info at https://www.man.eu/truckscenes
    """

    _target: Type = field(default_factory=lambda: TruckScenes)
    """target class to instantiate"""
    sequence: str = "a6a87db5125846bda72ccfc9931ee153-11"
    """Name of the scene."""
    data: Path = Path("data/truckscenes")
    """Path to NuScenes dataset."""
    version: Literal["v1.0-mini", "v1.0-trainval"] = "v1.0-mini"
    """Dataset version."""
    cameras: Tuple[
        Literal[
            "LEFT_FRONT",
            "RIGHT_FRONT",
            "LEFT_BACK",
            "RIGHT_BACK",
            "none",
            "all",
        ],
        ...,
    ] = ("all",)
    """Which cameras to use."""
    lidars: Tuple[
        Literal[
            "LEFT",  # pandar64
            "RIGHT",  # pandar64
            "REAR",  # ouster
            "TOP_FRONT",  # ouster
            "TOP_LEFT",  # ouster
            "TOP_RIGHT",  # ouster
            "all",
            "none",
        ],
        ...,
    ] = (
        "LEFT",
        "RIGHT",
    )
    radars: Tuple[
        Literal[
            "LEFT_FRONT",
            "LEFT_BACK",
            "LEFT_SIDE",
            "RIGHT_FRONT",
            "RIGHT_BACK",
            "RIGHT_SIDE",
            "all",
            "none",
        ],
        ...,
    ] = ("none",)
    """Which lidars to use. Currently only supports LIDAR_TOP"""
    verbose: bool = False
    """Load dataset with verbose messaging"""
    annotation_interval: float = 0.5
    """Length of time interval used to sample annotations from the dataset"""
    include_deformable_actors: bool = True
    """Include deformable actors in the dataset (NuScenes has many pedestrians so we default this to true)."""
    train_eval_split_type: SplitTypes = SplitTypes.LINSPACE
    """Type of split to use for train/eval split."""
    lidar_elevation_mapping: Dict[str, Dict] = field(default_factory=lambda: TRUCKSCENES_ELEVATION_MAPPING)
    """Elevation mapping for each lidar."""
    skip_elevation_channels: Dict[str, Tuple] = field(default_factory=lambda: TRUCKSCENES_SKIP_ELEVATION_CHANNELS)
    """Channels to skip when adding missing points."""
    lidar_azimuth_resolution: Dict[str, float] = field(default_factory=lambda: TRUCKSCENES_AZIMUTH_RESOLUTION)
    """Azimuth resolution for each lidar."""
    add_missing_points: bool = False
    """Add missing points to lidar point clouds."""
    rolling_shutter_offsets: Tuple[float, float] = (0.0, 1 / 30.0)
    """The time offset for the first and last line, relative to the image timestamp (seconds)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if "scene" not in self.sequence:
            self.sequence = "scene-" + self.sequence


@dataclass
class TruckScenes(ADDataParser):
    """NuScenes DatasetParser"""

    config: TruckScenesDataParserConfig

    @property
    def actor_transform(self) -> torch.Tensor:
        """Nuscenes uses x-forward, so we need to rotate to x-right."""
        return torch.from_numpy(WLH_TO_LWH)

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        if "all" in self.config.cameras:
            self.config.cameras = AVAILABLE_CAMERAS
        filenames, times, intrinsics, poses, cam2egos, idxs = [], [], [], [], [], []
        heights, widths = [], []
        first_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        is_key_frame = []
        for cam_idx, camera in enumerate(["CAMERA_" + camera for camera in self.config.cameras]):
            for sample_data in self._find_all_sample_data(first_sample["data"][camera]):
                calibrated_sensor_data = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
                ego_pose_data = self.nusc.get("ego_pose", sample_data["ego_pose_token"])
                ego_pose = _rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])
                cam_pose = _rotation_translation_to_pose(
                    calibrated_sensor_data["rotation"], calibrated_sensor_data["translation"]
                )
                cam_pose[:3, :3] = cam_pose[:3, :3] @ OPENCV_TO_NERFSTUDIO
                pose = ego_pose @ cam_pose
                cam2egos.append(cam_pose)
                poses.append(pose)
                filenames.append(self.config.data / sample_data["filename"])
                intrinsics.append(calibrated_sensor_data["camera_intrinsic"])
                times.append(sample_data["timestamp"] / 1e6)
                idxs.append(cam_idx)
                heights.append(
                    DEFAULT_IMAGE_HEIGHT - CAMERA_TO_BOTTOM_RIGHT_CROP[camera[7:]][0]
                )  # :4 to remove CAM_
                widths.append(
                    DEFAULT_IMAGE_WIDTH - CAMERA_TO_BOTTOM_RIGHT_CROP[camera[7:]][1]
                )  # :4 to remove CAM_
                is_key_frame.append(sample_data["is_key_frame"])

        # To tensors
        intrinsics = torch.tensor(np.array(intrinsics), dtype=torch.float32)
        poses = torch.tensor(np.array(poses), dtype=torch.float32)
        cam2egos = torch.tensor(np.array(cam2egos), dtype=torch.float32)
        times = torch.tensor(times, dtype=torch.float64)
        idxs = torch.tensor(idxs).int().unsqueeze(-1)
        heights = torch.tensor(heights).int()
        widths = torch.tensor(widths).int()
        is_key_frame = torch.tensor(is_key_frame).reshape(-1, 1).bool()
        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            height=heights,
            width=widths,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            times=times,
            metadata={"sensor_idxs": idxs, "is_key_frame": is_key_frame},
        )
        return cameras, filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        lidar_filenames, times, poses, lid2egos, idxs = [], [], [], [], []
        first_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        is_key_frame = []
        if "all" in self.config.lidars:
            self.config.lidars = AVAILABLE_LIDARS
        
        for lidar_idx, lidar in enumerate(["LIDAR_" + lidar for lidar in self.config.lidars]):
            for lidar_data in self._find_all_sample_data(first_sample["data"][lidar]):
                calibrated_sensor_data = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
                ego_pose_data = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
                ego_pose = _rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])
                lidar_pose = _rotation_translation_to_pose(
                    calibrated_sensor_data["rotation"], calibrated_sensor_data["translation"]
                )
                pose = ego_pose @ lidar_pose

                lidar_filenames.append(self.config.data / lidar_data["filename"])

                poses.append(pose)
                lid2egos.append(lidar_pose)
                times.append(lidar_data["timestamp"] / 1e6)
                idxs.append(lidar_idx)
                is_key_frame.append(lidar_data["is_key_frame"])

        poses = torch.tensor(np.array(poses), dtype=torch.float64)  # will be changed to float32 later
        lid2egos = torch.tensor(np.array(lid2egos), dtype=torch.float32)
        times = torch.tensor(times, dtype=torch.float64)  # need higher precision
        idxs = torch.tensor(idxs).int().unsqueeze(-1)

        is_key_frame = torch.tensor(is_key_frame).reshape(-1, 1).bool()
        lidars = Lidars(
            lidar_to_worlds=poses[:, :3, :4],
            lidar_type=LidarType.VELODYNE_HDL32E,
            assume_ego_compensated=True,
            times=times,
            metadata={"sensor_idxs": idxs, "is_key_frame": is_key_frame},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
            valid_lidar_distance_threshold=DUMMY_DISTANCE_VALUE / 2,
        )
        return lidars, lidar_filenames

    def _read_lidars(self, lidars: Lidars, filepaths: List[Path]) -> List[torch.Tensor]:
        point_clouds = []
        for filepath in filepaths:
            pc = pypcd4.PointCloud.from_path(str(filepath))
            xyz = pc.numpy(("x", "y", "z"))
            # we might want to filter some of the points here somehow. We can get returns on the ego trailer, which we don't want.
            reflectance = pc.numpy(("intensity",)) # already normalized
            timestamp = pc.numpy(("timestamp",)).astype(np.int64)
            pc_timestamp = int(filepath.stem.split("_")[-1])
            timestamp -= pc_timestamp # TODO: is this how we want it, i think so

            pc = np.concatenate(
                [
                    xyz,
                    reflectance,  # add reflectance as last channel
                    timestamp.astype(np.int32),  # add timestamp as last channel
                ],
                axis=-1,
            )

            # TODO: add the channel info. we probably need to infer it similar to PandaSetDataParser...

            point_clouds.append(torch.from_numpy(pc))

        if self.config.add_missing_points:
            # the points are not ego-motion compensated (good), so we dont have to do that.
            # TODO: add missing points
            raise NotImplementedError(
                "Adding missing points is not implemented for TruckScenes yet. Please set add_missing_points=False."
            )
            # remove ego motion compensation
            poses = lidars.lidar_to_worlds
            times = lidars.times.squeeze(-1)
            missing_points = []
            for point_cloud, l2w, time in zip(point_clouds, poses, times):
                pc = point_cloud.clone()
                # absolute time
                pc[:, 4] = pc[:, 4] + time
                # project to world frame
                pc[..., :3] = transform_points(pc[..., :3], l2w.unsqueeze(0).to(pc))
                # remove ego motion compensation
                pc, interpolated_poses = self._remove_ego_motion_compensation(pc, poses, times)
                # reset time
                pc[:, 4] = point_cloud[:, 4].clone()
                # transform to common lidar frame again
                interpolated_poses = torch.matmul(
                    pose_utils.inverse(l2w.unsqueeze(0)).float(), pose_utils.to4x4(interpolated_poses).float()
                )
                # move channel from index 5 to 3
                pc = pc[..., [0, 1, 2, 5, 3, 4]]
                # add missing points
                missing_points.append(self._get_missing_points(pc, interpolated_poses, "LIDAR_TOP", dist_cutoff=0.05))

            # add missing points to point clouds
            point_clouds = [torch.cat([pc, missing], dim=0) for pc, missing in zip(point_clouds, missing_points)]
        # we do this here as we want to have the poses in float64 for the ego motion compensation removal 
        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()
        return point_clouds

    def _generate_dataparser_outputs(self, split="train"):
        self.nusc = TruckScenesDatabase(
            version=self.config.version,
            dataroot=str(self.config.data.absolute()),
            verbose=self.config.verbose,
        )
        self.scene = self.nusc.get("scene", self.nusc.field2token("scene", "name", str(self.config.sequence))[0])
        out = super()._generate_dataparser_outputs(split)
        del self.nusc
        del self.scene
        return out

    def _find_all_sample_data(self, sample_data_token: str):
        """Finds all sample data from a given sample token."""
        curr_token = sample_data_token
        sd = self.nusc.get("sample_data", curr_token)
        # Rewind to first sample data
        while sd["prev"]:
            curr_token = sd["prev"]
            sd = self.nusc.get("sample_data", curr_token)
        # Forward to last sample data
        all_sample_data = [sd]
        while sd["next"]:
            curr_token = sd["next"]
            sd = self.nusc.get("sample_data", curr_token)
            all_sample_data.append(sd)
        return all_sample_data

    def _get_actor_trajectories(self) -> List[Dict]:
        trajs = defaultdict(list)
        curr_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        while True:
            for box_token in curr_sample["anns"]:
                box = self.nusc.get_box(box_token)
                pose = np.eye(4)
                pose[:3, :3] = box.orientation.rotation_matrix
                pose[:3, 3] = np.array(box.center)
                pose = pose @ WLH_TO_LWH
                instance_token = self.nusc.get("sample_annotation", box.token)["instance_token"]
                trajs[instance_token].append(
                    {
                        "pose": pose,
                        "wlh": np.array(box.wlh),
                        "label": box.name,
                        "time": curr_sample["timestamp"] / 1e6,
                    }
                )
            if curr_sample["next"]:
                curr_sample = self.nusc.get("sample", curr_sample["next"])
            else:
                break
        return self._traj_dict_to_list(trajs)

    def _traj_dict_to_list(self, traj: dict) -> list:
        """Convert a dictionary of lists with trajectories to a list of dictionaries with trajectories"""
        allowed_classes: Set[str] = set(ALLOWED_RIGID_CLASSES)
        if self.config.include_deformable_actors:
            allowed_classes.update(ALLOWED_DEFORMABLE_CLASSES)
        traj_out = []
        for instance_token, traj_list in traj.items():
            poses = torch.from_numpy(np.stack([t["pose"] for t in traj_list]).astype(np.float32))
            times = torch.from_numpy(np.array([t["time"] for t in traj_list]))
            dims = torch.from_numpy(np.array([t["wlh"] for t in traj_list]).astype(np.float32))
            dims = dims.max(0).values  # take max dimensions (important for deformable objects)
            dynamic = (poses[:, :2, 3].std(dim=0) > 0.50).any()
            stationary = not dynamic  # TODO: maybe make this stricter
            if stationary or not _is_label_allowed(traj_list[0]["label"], allowed_classes):
                continue
            traj_dict = {
                "uuid": instance_token,
                "label": traj_list[0]["label"],
                "poses": poses,
                "timestamps": times,
                "dims": dims,
                "stationary": stationary,
                "symmetric": "human" not in traj_list[0]["label"],
                "deformable": "human" in traj_list[0]["label"],
            }
            traj_out.append(traj_dict)
        return traj_out


def _rotation_translation_to_pose(r_quat, t_vec):
    """Convert quaternion rotation and translation vectors to 4x4 matrix"""

    pose = np.eye(4)

    # NB: Nuscenes recommends pyquaternion, which uses scalar-first format (w x y z)
    # https://github.com/nutonomy/nuscenes-devkit/issues/545#issuecomment-766509242
    # https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L299
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    pose[:3, :3] = pyquaternion.Quaternion(r_quat).rotation_matrix

    pose[:3, 3] = t_vec
    return pose

def _is_label_allowed(label: str, allowed_classes: Set[str]) -> bool:
    """Check if label is allowed, on all possible hierarchies."""
    split_label = label.split(".")
    for i in range(len(split_label)):
        if ".".join(split_label[: i + 1]) in allowed_classes:
            return True
    return False


if __name__ == "__main__":
    config = TruckScenesDataParserConfig()
    config.data = Path("data/man-truckscenes")
    config.lidars = ("REAR",)

    tsc = TruckScenes(config=config)
    out = tsc._generate_dataparser_outputs(split="train")


    idx = 0
    all_elevations = set()
    # from plotly import graph_objects as go

    # fig = go.Figure()
    # # plot the point cloud
    # xyz = out.metadata["point_clouds"][50]
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=xyz[:, 0],
    #         y=xyz[:, 1],
    #         z=xyz[:, 2],
    #         mode="markers",
    #         marker=dict(size=1, color=xyz[:, 3], colorscale="Viridis", opacity=0.8),
    #     )
    # )
    # fig.update_layout(
    #     scene=dict(
    #         xaxis_title="X",
    #         yaxis_title="Y",
    #         zaxis_title="Z",
    #         aspectmode="data",
    #     ),
    #     title=f"Lidar {idx} Point Cloud",
    # )
    # fig.show()

    for pcs in out.metadata["point_clouds"]:
        # lets see if we can find th

        # convert to torch tensor
        dist = torch.norm(pcs[:, :3], dim=-1)
        elevation = torch.arcsin(pcs[:, 2] / dist)
        elevation = torch.rad2deg(elevation)

        # make unique up to 4 decimal places
        elevation = torch.round(elevation * 1e4) / 1e4
        elevation = elevation.unique()
        # add
        all_elevations.update(elevation.tolist())

        # make horizontal bins 
        azimuth = torch.atan2(pcs[:, 1], pcs[:, 0])
        # plot the diff
        import matplotlib.pyplot as plt
        plt.hist(np.diff(sorted(azimuth.numpy())), bins=100, alpha=0.5, label=f"Lidar {idx} Azimuth")
        plt.savefig(f"lidar_{idx}_elevation_histogram.png")   

    print(f"Total unique elevations: {len(all_elevations)}")
    


    # histc, bin_edges = torch.histogram(elevation, bins=30000)

    # import matplotlib.pyplot as plt
    # plt.plot(bin_edges[:-1], histc.numpy())
    # plt.title(f"Lidar {lidars.lidar_type} elevation histogram")
    # plt.xlabel("Elevation (degrees)")
    # plt.ylabel("Count")
    # plt.savefig("lidar_elevation_histogram.png")
    



