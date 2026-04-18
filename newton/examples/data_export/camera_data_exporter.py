# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Camera data exporter for multi-camera simulations.

This module provides utilities to export color and depth data from camera sensors
in Newton simulations. It can work with any simulation model without requiring
modifications to the original example code.

Usage:
    >>> from newton.examples.data_export.camera_data_exporter import CameraDataExporter
    >>> 
    >>> # Create exporter with model and state from your simulation
    >>> exporter = CameraDataExporter(
    ...     model=your_model,
    ...     state=your_state,
    ...     camera_configs=[
    ...         {"pos": wp.vec3(0, 0, 1.5), "pitch": -90, "yaw": 0, "fov": 60},
    ...         {"pos": wp.vec3(1, 0.8, 0.5), "pitch": -30, "yaw": -135, "fov": 50},
    ...     ],
    ...     camera_names=["overhead", "front"],
    ...     output_dir="./camera_export"
    ... )
    >>>
    >>> # Export data for each frame
    >>> for frame_idx in range(100):
    ...     exporter.render_frame(your_state, frame_idx)
"""

from __future__ import annotations

import json
import math
import numpy as np
import warp as wp
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING

from newton.sensors import SensorTiledCamera

if TYPE_CHECKING:
    from newton import Model, State


class CameraDataExporter:
    """Export color and depth data from multiple camera views.
    
    This class manages rendering and exporting of camera sensor data including
    color images, depth maps, and camera parameters from multiple viewpoints
    in a Newton simulation.
    
    Attributes:
        model: The Newton model to render.
        state: The simulation state.
        camera_configs: List of camera configuration dicts with pos, pitch, yaw, fov.
        camera_names: Names for each camera view.
        output_dir: Directory where exported data will be saved.
    """
    
    def __init__(
        self,
        model: Model,
        state: State,
        camera_configs: list[dict] | None = None,
        camera_names: list[str] | None = None,
        output_dir: str = "./camera_export",
        render_width: int = 640,
        render_height: int = 480,
    ):
        """Initialize the camera data exporter.
        
        Args:
            model: The Newton model to render from.
            state: The simulation state.
            camera_configs: List of dicts with keys: pos (wp.vec3), pitch, yaw, fov.
                If None, uses a default single camera.
            camera_names: Human-readable names for each camera. If None, uses camera_0, camera_1, etc.
            output_dir: Root directory for storing exported data.
            render_width: Rendering resolution width in pixels.
            render_height: Rendering resolution height in pixels.
        """
        self.model = model
        self.state = state
        self.output_dir = Path(output_dir)
        self.render_width = render_width
        self.render_height = render_height
        
        # Default to single camera if not specified
        if camera_configs is None:
            camera_configs = [{
                "pos": wp.vec3(10.0, 0.0, 2.0),
                "pitch": -5.0,
                "yaw": -45.0,
                "fov": 45.0
            }]
        
        self.camera_configs = camera_configs
        self.num_cameras = len(camera_configs)
        
        if camera_names is None:
            self.camera_names = [f"camera_{i}" for i in range(self.num_cameras)]
        else:
            if len(camera_names) != self.num_cameras:
                raise ValueError(
                    f"camera_names length ({len(camera_names)}) must match "
                    f"num_cameras ({self.num_cameras})"
                )
            self.camera_names = camera_names
        
        # Create output directories
        for name in self.camera_names:
            color_dir = self.output_dir / name / "color"
            depth_dir = self.output_dir / name / "depth"
            color_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup sensor
        self.sensor = SensorTiledCamera(
            model=model,
            config=SensorTiledCamera.Config(
                default_light=True,
                default_light_shadows=True,
                checkerboard_texture=True,
                backface_culling=True,
            ),
        )
        
        # Compute camera rays (same for all cameras in pinhole projection)
        fov_rad = math.radians(camera_configs[0]["fov"])
        self.camera_rays = self.sensor.compute_pinhole_camera_rays(
            render_width, render_height, fov_rad
        )
        
        # Create output images for each camera
        self.color_images = [
            self.sensor.create_color_image_output(render_width, render_height, 1)
            for _ in range(self.num_cameras)
        ]
        self.depth_images = [
            self.sensor.create_depth_image_output(render_width, render_height, 1)
            for _ in range(self.num_cameras)
        ]
        self.shape_index_images = [
            self.sensor.create_shape_index_image_output(render_width, render_height, 1)
            for _ in range(self.num_cameras)
        ]
    
    def camera_config_to_transform(self, config: dict) -> wp.transform:
        """Convert camera config (pos, pitch, yaw, fov) to world transform.
        
        Args:
            config: Dict with keys: pos (wp.vec3), pitch, yaw (degrees).
        
        Returns:
            World-to-camera transform for rendering.
        """
        pos = config.get("pos", wp.vec3(0.0, 0.0, 0.0))
        pitch_deg = config.get("pitch", 0.0)
        yaw_deg = config.get("yaw", 0.0)
        
        # Convert degrees to radians
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        
        # Create rotation from pitch and yaw (ZYX Euler angles)
        # Pitch is rotation around X axis (up/down)
        # Yaw is rotation around Z axis (left/right)
        quat_pitch = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), pitch)
        quat_yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
        quat = quat_yaw * quat_pitch  # Combine rotations
        
        return wp.transform(pos, quat)
    
    def render_frame(self, state: State, frame_idx: int) -> None:
        """Render and export data for all cameras at a given frame.
        
        Args:
            state: The simulation state at this frame.
            frame_idx: Frame index for naming output files.
        """
        # Sync transforms from state
        self.sensor.sync_transforms(state)
        
        # Render from each camera
        for cam_id in range(self.num_cameras):
            config = self.camera_configs[cam_id]
            
            # Get camera transform from config
            cam_transform = self.camera_config_to_transform(config)
            camera_transforms = wp.array([[cam_transform]], dtype=wp.transform)
            
            # Render
            self.sensor.update(
                state,
                camera_transforms,
                self.camera_rays,
                color_image=self.color_images[cam_id],
                depth_image=self.depth_images[cam_id],
                shape_index_image=self.shape_index_images[cam_id],
            )
            
            # Export color and depth for this camera
            self._export_camera_frame(cam_id, frame_idx)
    
    def _export_camera_frame(self, camera_id: int, frame_idx: int) -> None:
        """Export color and depth images for a single camera.
        
        Args:
            camera_id: Index of the camera.
            frame_idx: Frame index for naming.
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL required for image export. Install with: pip install pillow")
        
        camera_name = self.camera_names[camera_id]
        color_image = self.color_images[camera_id]
        depth_image = self.depth_images[camera_id]
        
        # Export color image
        color_np = color_image.numpy()[0, 0]  # (H, W) as uint32 packed RGBA
        color_path = self.output_dir / camera_name / "color" / f"{frame_idx:05d}.png"
        
        # Convert packed RGBA to RGB image
        color_rgb = self._unpack_color(color_np)
        Image.fromarray(color_rgb, mode="RGB").save(str(color_path))
        
        # Export depth image as NPY
        depth_np = depth_image.numpy()[0, 0]  # (H, W) as float32
        depth_path = self.output_dir / camera_name / "depth" / f"{frame_idx:05d}.npy"
        np.save(str(depth_path), depth_np)
        
        print(f"Saved frame {frame_idx} from {camera_name}: color={color_path.name}, depth={depth_path.name}")
    
    def _unpack_color(self, packed_rgba: np.ndarray) -> np.ndarray:
        """Convert packed uint32 RGBA to RGB uint8 array.
        
        Args:
            packed_rgba: Array of uint32 values with RGBA packed as BGRA in uint32.
        
        Returns:
            Array of shape (H, W, 3) with RGB values.
        """
        h, w = packed_rgba.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Unpack BGRA from uint32
        rgb[:, :, 0] = (packed_rgba >> 16) & 0xFF  # R
        rgb[:, :, 1] = (packed_rgba >> 8) & 0xFF   # G
        rgb[:, :, 2] = (packed_rgba >> 0) & 0xFF   # B
        
        return rgb
    
    def export_camera_info(self, frame_idx: int) -> None:
        """Export camera intrinsics and extrinsics as JSON.
        
        Args:
            frame_idx: Frame index to include in output.
        """
        camera_info = {
            "frame": frame_idx,
            "cameras": []
        }
        
        for cam_id, config in enumerate(self.camera_configs):
            intrinsics = self._compute_intrinsics(config["fov"])
            
            camera_info["cameras"].append({
                "name": self.camera_names[cam_id],
                "intrinsics": intrinsics,
                "extrinsics": {
                    "position": [float(config["pos"][0]), float(config["pos"][1]), float(config["pos"][2])],
                    "pitch": config["pitch"],
                    "yaw": config["yaw"],
                }
            })
        
        info_path = self.output_dir / f"camera_info_{frame_idx:05d}.json"
        with open(info_path, "w") as f:
            json.dump(camera_info, f, indent=2)
    
    def _compute_intrinsics(self, fov_degrees: float) -> dict:
        """Compute camera intrinsic matrix parameters.
        
        Args:
            fov_degrees: Field of view in degrees.
        
        Returns:
            Dict with fx, fy, cx, cy (in pixels).
        """
        fov_rad = math.radians(fov_degrees)
        fx = self.render_width / (2 * math.tan(fov_rad / 2))
        fy = self.render_height / (2 * math.tan(fov_rad / 2))
        cx = self.render_width / 2.0
        cy = self.render_height / 2.0
        
        return {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": self.render_width,
            "height": self.render_height,
        }
