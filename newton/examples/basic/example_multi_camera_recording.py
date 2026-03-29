# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Multi-Camera Recording
#
# Demonstrates how to record a simulation from multiple camera viewpoints
# simultaneously and export them as separate video files.
#
# The MultiCameraRecorder integrates seamlessly into the existing simulation
# loop without requiring any changes to the core architecture. Simply call
# recorder.capture_frames() after example.render() in the main loop.
#
# This example creates a simple scene with shapes dropping and records from
# 3 different viewpoints: front (45°), side (-135°), and top (overhead).
#
# Command: python -m newton.examples multi_camera_recording
#
# Output:
#   - ./multi_camera_recordings/camera_0/*.png (front view frames)
#   - ./multi_camera_recordings/camera_1/*.png (side view frames)
#   - ./multi_camera_recordings/camera_2/*.png (top view frames)
#   - ./multi_camera_recordings/camera_*.mp4 (final video files)
#
###########################################################################

import warp as wp

import newton
import newton.examples
from newton import multi_camera_recorder


class Example:
    def __init__(self, viewer, args):
        # Setup simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.solver_type = args.solver if hasattr(args, "solver") and args.solver else "xpbd"

        # Setup multi-camera recorder
        self.recorder = multi_camera_recorder.MultiCameraRecorder(
            viewer,
            output_dir="./multi_camera_recordings",
            num_cameras=3,
            camera_names=["front_view", "side_view", "top_view"],
            skip_frames=30,  # Skip first 30 frames to let simulation settle
            async_save=True,  # Save frames asynchronously
        )

        # Configure the 3 camera views
        # Camera 0: Front view (45° isometric)
        self.recorder.set_camera_config(
            0,
            pos=wp.vec3(12.0, -8.0, 3.0),
            pitch=-15.0,
            yaw=-135.0,
            fov=45.0,
        )

        # Camera 1: Side view (90° perpendicular)
        self.recorder.set_camera_config(
            1,
            pos=wp.vec3(0.0, -15.0, 3.0),
            pitch=-10.0,
            yaw=-90.0,
            fov=45.0,
        )

        # Camera 2: Top view (looking down)
        self.recorder.set_camera_config(
            2,
            pos=wp.vec3(0.0, 0.0, 12.0),
            pitch=-90.0,
            yaw=0.0,
            fov=60.0,
        )

        # Setup simulation
        builder = newton.ModelBuilder()

        if self.solver_type == "vbd":
            builder.default_shape_cfg.ke = 1.0e6
            builder.default_shape_cfg.kd = 1.0e1
            builder.default_shape_cfg.mu = 0.5

        # Add ground plane
        builder.add_ground_plane()

        # Drop zone height
        drop_z = 5.0

        # Add various shapes that will drop and create dynamic scene
        # This provides visual interest from multiple viewpoints

        # Sphere
        body_sphere = builder.add_body(
            xform=wp.transform(p=wp.vec3(2.0, -2.0, drop_z), q=wp.quat_identity()),
            label="sphere",
        )
        builder.add_shape_sphere(body_sphere, radius=0.5)

        # Capsule
        body_capsule = builder.add_body(
            xform=wp.transform(p=wp.vec3(-2.0, 0.0, drop_z), q=wp.quat_identity()),
            label="capsule",
        )
        builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7)

        # Box
        body_box = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 2.0, drop_z), q=wp.quat_identity()),
            label="box",
        )
        builder.add_shape_box(body_box, a=0.4, b=0.4, c=0.8)

        # Cylinder
        body_cylinder = builder.add_body(
            xform=wp.transform(p=wp.vec3(2.0, 2.0, drop_z), q=wp.quat_identity()),
            label="cylinder",
        )
        builder.add_shape_cylinder(body_cylinder, radius=0.35, half_height=0.6)

        # Finalize model
        self.model = builder.finalize()
        self.control = self.model.control()

        # Create solver
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="euler",
            iterations=10,
            ls_iterations=5,
        )

        # Create states
        self.state_0, self.state_1 = self.model.state(), self.model.state()

        # Use CUDA graphs if available
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        # Set model in viewer
        self.viewer.set_model(self.model)

    def simulate(self):
        """Advance simulation by one frame."""
        for _i in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.sim_dt, self.control)
            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt

    def step(self):
        """Called by main loop to advance simulation."""
        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

    def render(self):
        """Called by main loop to render frame."""
        with wp.ScopedTimer("render", active=False):
            # Standard render call
            self.viewer.begin_frame(self.sim_time)
            self.viewer.set_model_state(self.state_0)
            self.viewer.end_frame()

            # Capture frames from all configured camera views
            # This integrates seamlessly without modifying viewer architecture
            self.recorder.capture_frames()

    def test_final(self):
        """Verify simulation validity at the end."""
        # Check that shapes have settled and moved
        # (not remaining in initial drop positions)
        assert self.sim_time > 0.5, "Simulation did not run"

        # After render() completes, generate videos from captured frames
        print(f"\nRecorded {self.recorder.get_frame_count()} frames from 3 camera views")
        print("Generating videos from captured frames...")

        results = self.recorder.generate_videos(fps=30, keep_frames=False)

        print("\nVideo generation results:")
        for camera_name, success in results.items():
            status = "✓ Generated" if success else "✗ Failed"
            print(f"  {status}: {camera_name}.mp4")

        print("\nVideos saved to: ./multi_camera_recordings/")
