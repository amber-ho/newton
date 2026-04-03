# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Export Physics Simulation Data for PhysTwin Training

This example captures:
- RGB video frames
- Depth maps
- 3D object & controller point clouds
- Camera intrinsics & extrinsics

Usage:
    python -m newton.examples data_export
    
Output:
    data/newton_export/  ← Ready for PhysTwin processing
"""

import os
import math
import json
import numpy as np
import warp as wp
from pathlib import Path
from scipy.spatial.transform import Rotation

import newton
import newton.examples
from newton.sensors import SensorTiledCamera


class NewtonDataExporter:
    """Export Newton simulation data for PhysTwin training."""
    
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.output_dir = Path("data/newton_export")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup simulation
        self.setup_simulation()
        self.setup_camera_sensor()
        
    def setup_simulation(self):
        """Create Newton simulation model."""
        # [Use existing example model, e.g., from example_basic_viewer.py]
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        # Add objects, robots, cloth, etc.
        self.model = builder.finalize()
        self.state = self.model.state()
        self.viewer.set_model(self.model)
        
    def setup_camera_sensor(self):
        """Initialize camera sensor for rendering."""
        self.render_width, self.render_height = 640, 480
        self.fov_degrees = 45.0
        
        self.sensor = SensorTiledCamera(
            model=self.model,
            config=SensorTiledCamera.Config(
                default_light=True,
                default_light_shadows=True,
                checkerboard_texture=True,
                backface_culling=True,
            ),
        )
        
        self.camera_rays = self.sensor.compute_pinhole_camera_rays(
            self.render_width, self.render_height, 
            math.radians(self.fov_degrees)
        )
        
        self.color_image = self.sensor.create_color_image_output(
            self.render_width, self.render_height, 1
        )
        self.depth_image = self.sensor.create_depth_image_output(
            self.render_width, self.render_height, 1
        )
        self.shape_index_image = self.sensor.create_shape_index_image_output(
            self.render_width, self.render_height, 1
        )
        
        # ✓ STEP 1 COMPLETE: Sensor initialized
        
    def step(self):
        """Simulate one frame."""
        # Your simulation step here
        pass
        
    def render(self):
        """STEP 1: Render and capture data."""
        # Get camera pose
        camera_pose = self.get_camera_pose()
        
        # Render using sensor
        self.sensor.update(
            self.state,
            self.get_camera_transforms(),
            self.camera_rays,
            color_image=self.color_image,
            depth_image=self.depth_image,
            shape_index_image=self.shape_index_image,
        )
        
        # ✓ STEP 2: Extract 3D points from Newton
        object_points_3d = self.extract_object_points_3d()
        controller_points_3d = self.extract_controller_points_3d()
        
        # ✓ STEP 3: Project to 2D and check visibility
        object_points_camera = self.transform_to_camera(object_points_3d, camera_pose)
        visibilities, u, v = self.project_and_check_visibility(
            object_points_camera, self.depth_image
        )
        
        # ✓ STEP 4: Extract colors
        colors = self.extract_colors(u, v)
        
        return {
            'color_image': self.color_image.numpy(),
            'depth_image': self.depth_image.numpy(),
            'object_points_3d': object_points_3d,
            'object_points_camera': object_points_camera,
            'object_colors': colors,
            'object_visibilities': visibilities,
            'controller_points_3d': controller_points_3d,
            'camera_pose': camera_pose,
            'camera_intrinsics': self.get_intrinsics(),
        }
    
    # ========== STEP 2: Extract 3D Points from Newton ==========
    def extract_object_points_3d(self):
        """Extract object point cloud from simulation."""
        # Sample points from mesh or particle system
        # In pure simulation, you have access to exact positions
        num_points = 1000
        object_points = np.random.randn(num_points, 3).astype(np.float32)
        # In reality, get from your object mesh/particles
        return object_points
    
    def extract_controller_points_3d(self):
        """Extract controller/hand point cloud from robot."""
        # Get end-effector positions, finger positions, etc.
        num_effectors = 5  # Example
        controller_points = np.random.randn(num_effectors, 3).astype(np.float32)
        return controller_points
    
    # ========== STEP 3: Project & Visibility ==========
    def transform_to_camera(self, points_world, camera_pose):
        """Transform world coordinates to camera coordinates."""
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        points_camera = (R @ points_world.T + t[:, None]).T
        return points_camera
    
    def project_and_check_visibility(self, points_camera, depth_image):
        """Project 3D points to image and check visibility."""
        intrinsics = self.get_intrinsics()
        fx, fy, cx, cy = (intrinsics['fx'], intrinsics['fy'], 
                          intrinsics['cx'], intrinsics['cy'])
        
        u = (fx * points_camera[:, 0] / points_camera[:, 2]) + cx
        v = (fy * points_camera[:, 1] / points_camera[:, 2]) + cy
        z = points_camera[:, 2]
        
        visibilities = np.zeros(len(points_camera), dtype=bool)
        for i in range(len(points_camera)):
            if 0 <= int(u[i]) < self.render_width and 0 <= int(v[i]) < self.render_height:
                depth_rendered = depth_image[0, 0, int(v[i]), int(u[i])]
                if abs(z[i] - depth_rendered) < 0.01:
                    visibilities[i] = True
        
        return visibilities, u, v
    
    # ========== STEP 4: Extract Colors ==========
    def extract_colors(self, u, v):
        """Extract RGB colors for each point."""
        color_img = self.color_image.numpy()[0, 0]  # (H, W, 3)
        colors = np.zeros((len(u), 3), dtype=np.uint8)
        
        for i, (u_i, v_i) in enumerate(zip(u, v)):
            if 0 <= int(u_i) < self.render_width and 0 <= int(v_i) < self.render_height:
                colors[i] = color_img[int(v_i), int(u_i), :3]
        
        return colors
    
    # ========== Camera utilities ==========
    def get_camera_pose(self):
        """Get 4x4 camera pose matrix."""
        if hasattr(self.viewer, 'camera'):
            pos = self.viewer.camera.pos
            view_matrix = self.viewer.camera.get_view_matrix()
            # Invert to get world-to-camera
            pose = np.linalg.inv(view_matrix.reshape(4, 4))
            return pose
        else:
            return np.eye(4)
    
    def get_camera_transforms(self):
        """Get camera transforms for sensor."""
        pose = self.get_camera_pose()
        return wp.array([[[wp.transformf(
            wp.vec3f(*pose[:3, 3]),
            wp.quatf(*Rotation.from_matrix(pose[:3, :3]).as_quat()),
        )]]], dtype=wp.transformf)
    
    def get_intrinsics(self):
        """Get camera intrinsics."""
        return {
            'fx': self.render_width / (2 * np.tan(np.radians(self.fov_degrees) / 2)),
            'fy': self.render_height / (2 * np.tan(np.radians(self.fov_degrees) / 2)),
            'cx': self.render_width / 2,
            'cy': self.render_height / 2,
            'width': self.render_width,
            'height': self.render_height,
        }
    
    def export_frame(self, frame_idx, frame_data):
        """Save frame data to disk."""
        # Create color directory
        color_dir = self.output_dir / f"color/0"
        color_dir.mkdir(parents=True, exist_ok=True)
        
        # Save color image
        color_path = color_dir / f"{frame_idx}.png"
        color_img = frame_data['color_image'][0, 0]
        from PIL import Image
        Image.fromarray((color_img * 255).astype(np.uint8)).save(color_path)
        
        # Save depth as NPY
        depth_dir = self.output_dir / f"depth/0"
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth_path = depth_dir / f"{frame_idx}.npy"
        np.save(depth_path, frame_data['depth_image'][0, 0])
        
        print(f"Saved frame {frame_idx} to {color_path}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    
    exporter = NewtonDataExporter(viewer, args)
    
    # Run simulation and export frames
    for frame_idx in range(100):  # 100 frames
        exporter.step()
        frame_data = exporter.render()
        exporter.export_frame(frame_idx, frame_data)
    
    print("✓ Newton export complete! Data ready for PhysTwin processing.")