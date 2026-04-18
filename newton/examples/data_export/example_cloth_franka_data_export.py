# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example: Export camera data from cloth_franka simulation.

This example demonstrates how to integrate camera data export
with the cloth_franka simulation. The MultiCameraRecorder writes RGB
frames directly and now also exports depth maps as `.npy` files.

Usage:
    python -m newton.examples cloth_franka_with_data_export

Output:
    ./cloth_franka_recordings/      → Video files, PNG frames, and depth NPY files
    ./cloth_franka_data_export/     → Copied color images, depth maps, camera info
"""

from __future__ import annotations

import numpy as np
import warp as wp
from pathlib import Path
from PIL import Image
import json
import time

import newton
import newton.examples
from newton.examples.cloth.example_cloth_franka import Example as ClothFrankaExample


class ClothFrankaDataExportExample:
    """Cloth Franka example with integrated camera data export."""
    
    def __init__(self, viewer, args):
        """Initialize the cloth franka example with data export.
        
        Args:
            viewer: Newton viewer instance.
            args: Command-line arguments.
        """
        # Create the base cloth_franka example (includes MultiCameraRecorder)
        self.cloth_franka = ClothFrankaExample(viewer, args)
        
        # Get model from cloth_franka
        self.model = self.cloth_franka.model
        self.viewer = viewer
        
        # Setup camera info matching cloth_franka's 3 cameras
        self._setup_camera_info()
        
        # Output directory for extracted color/depth data
        self.output_dir = Path("./cloth_franka_data_export")
        self._setup_output_dirs()
        
        self.frame_count = 0
        self._export_start_frame = 60  # Skip first 60 frames like cloth_franka does
        self._export_frequency = 1  # Export every frame (or use > 1 to skip frames)
    
    def _setup_camera_info(self) -> None:
        """Setup camera information matching cloth_franka's recorder."""
        # Camera configs matching cloth_franka's recorder setup
        self.camera_configs = [
            {
                "name": "overhead",
                "pos": (0.0, 0.0, 1.5),
                "pitch": -90.0,  # Looking straight down
                "yaw": 0.0,
                "fov": 60.0,
            },
            {
                "name": "front",
                "pos": (1.0, 0.8, 0.5),
                "pitch": -30.0,
                "yaw": -135.0,
                "fov": 50.0,
            },
            {
                "name": "side",
                "pos": (0.0, 1.2, 0.7),
                "pitch": -25.0,
                "yaw": -90.0,
                "fov": 50.0,
            },
        ]
    
    def _setup_output_dirs(self) -> None:
        """Create output directories for each camera."""
        for camera_config in self.camera_configs:
            name = camera_config["name"]
            color_dir = self.output_dir / name / "color"
            depth_dir = self.output_dir / name / "depth"
            color_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
    
    def step(self):
        """Step the simulation."""
        self.cloth_franka.step()
    
    def render(self):
        """Render and optionally export camera data."""
        self.cloth_franka.render()
        
        # Export camera data after warmup period and at specified frequency
        should_export = (
            self.frame_count >= self._export_start_frame and
            (self.frame_count - self._export_start_frame) % self._export_frequency == 0
        )
        
        if should_export:
            export_frame_idx = self.frame_count - self._export_start_frame
            print(f"\n--- Processing camera data for frame {export_frame_idx} (sim frame {self.frame_count}) ---")
            
            # Extract color/depth exported by MultiCameraRecorder
            self._process_recorded_frames(export_frame_idx)
            
            # Export camera intrinsics/extrinsics info periodically
            if export_frame_idx % 100 == 0:
                self._export_camera_info(export_frame_idx)
        
        self.frame_count += 1
    
    def _process_recorded_frames(self, export_frame_idx: int) -> None:
        """Process frames saved by MultiCameraRecorder.
        
        This reads the PNG and NPY files saved by the recorder and copies them
        to the layout used by downstream training code.
        """
        recorder_dir = Path("./cloth_franka_recordings")
        
        # Wait for async save threads to finish writing to disk
        # This is necessary since the recorder uses async_save=True by default
        time.sleep(0.15)
        
        # The recorder saves frames with skip_frames offset already applied,
        # so the frame index matches our export_frame_idx directly
        for camera_config in self.camera_configs:
            camera_name = camera_config["name"]
            
            # Read the frame from the recorder
            frame_path = recorder_dir / camera_name / f"{export_frame_idx:05d}.png"
            
            if not frame_path.exists():
                # Frame may not exist yet if async threads haven't finished
                return
            
            depth_path = recorder_dir / camera_name / "depth" / f"{export_frame_idx:05d}.npy"

            if not depth_path.exists():
                return

            # Load the color image with retries for partially-written files
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    color_img = Image.open(frame_path)
                    color_img.load()
                    output_color_path = self.output_dir / camera_name / "color" / f"{export_frame_idx:05d}.png"
                    color_img.save(str(output_color_path))

                    depth_array = np.load(depth_path)
                    output_depth_path = self.output_dir / camera_name / "depth" / f"{export_frame_idx:05d}.npy"
                    np.save(output_depth_path, depth_array)

                    print(f"  ✓ Exported {camera_name} color frame: {output_color_path.name}")
                    print(f"  ✓ Exported {camera_name} depth frame: {output_depth_path.name}")
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.05)  # Wait before retry
                    else:
                        print(f"  Warning: Could not process {camera_name}: {e}")
    
    def _export_camera_info(self, frame_idx: int) -> None:
        """Export camera intrinsics and extrinsics as JSON.
        
        Args:
            frame_idx: Frame index to include in output.
        """
        camera_info = {
            "frame": frame_idx,
            "cameras": []
        }
        
        for camera_config in self.camera_configs:
            # Compute intrinsics for this camera's FOV
            fov_degrees = camera_config["fov"]
            fov_rad = np.radians(fov_degrees)
            render_width, render_height = 1024, 768  # Standard resolution
            fx = render_width / (2 * np.tan(fov_rad / 2))
            fy = render_height / (2 * np.tan(fov_rad / 2))
            cx = render_width / 2.0
            cy = render_height / 2.0
            
            camera_info["cameras"].append({
                "name": camera_config["name"],
                "intrinsics": {
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": float(cx),
                    "cy": float(cy),
                    "width": render_width,
                    "height": render_height,
                },
                "extrinsics": {
                    "position": camera_config["pos"],
                    "pitch": camera_config["pitch"],
                    "yaw": camera_config["yaw"],
                }
            })
        
        info_path = self.output_dir / f"camera_info_{frame_idx:05d}.json"
        with open(info_path, "w") as f:
            json.dump(camera_info, f, indent=2)
        
        print(f"  ✓ Exported camera info to {info_path.name}")
    
    def test_final(self):
        """Test final state (delegated to cloth_franka)."""
        self.cloth_franka.test_final()
    
    def test_post_step(self):
        """Test after each step (delegated to cloth_franka)."""
        if hasattr(self.cloth_franka, 'test_post_step'):
            self.cloth_franka.test_post_step()



if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=3850)
    viewer, args = newton.examples.init(parser)
    
    # Create example with integrated data export
    example = ClothFrankaDataExportExample(viewer, args)
    
    # Run the simulation
    newton.examples.run(example, args)
    
    print("\n✓ Cloth Franka simulation with data export complete!")
    print("  Videos: ./cloth_franka_recordings/")
    print("  Camera data: ./cloth_franka_data_export/")
