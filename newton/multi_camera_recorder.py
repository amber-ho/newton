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

"""Multi-camera recording support for Newton simulations.

This module provides functionality to record multiple camera views during a simulation
and export them as separate video files. Recording cameras are independent and do not
affect the visible simulation view.

The visible camera remains fixed on its current view while 3+ independent recording
cameras capture frames simultaneously from different angles. All views are recorded
without any on-screen camera jumping.

Example:
    >>> from newton import viewer, multi_camera_recorder
    >>> import warp as wp
    >>>
    >>> # Create viewer and recorder
    >>> v = viewer.ViewerGL()
    >>> recorder = multi_camera_recorder.MultiCameraRecorder(v, output_dir="./recordings")
    >>>
    >>> # Configure independent recording camera views
    >>> recorder.set_camera_config(0, pos=wp.vec3(10, 0, 2), pitch=-5, yaw=-45)
    >>> recorder.set_camera_config(1, pos=wp.vec3(-10, 0, 2), pitch=-5, yaw=135)
    >>> recorder.set_camera_config(2, pos=wp.vec3(0, 10, 2), pitch=-5, yaw=-135)
    >>>
    >>> # During simulation loop, capture frames from all cameras
    >>> # The displayed view stays fixed; recording happens off-screen
    >>> for frame in range(num_frames):
    ...     example.step()
    ...     example.render()
    ...     recorder.capture_frames()  # Off-screen capture from 3 camera views
    >>>
    >>> # Generate video files after simulation
    >>> recorder.generate_videos(fps=30)
"""

from __future__ import annotations

import glob
import os
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from collections.abc import Callable

    from . import Model, State
    from ._src.viewer import ViewerGL

__all__ = ["MultiCameraRecorder"]


class MultiCameraRecorder:
    """Records simulation frames from multiple independent camera viewpoints.

    This class manages capture of simulation frames from multiple camera configurations,
    storing them separately for later video generation. It hooks into the existing
    simulation loop without modifying the viewer's core architecture.

    Attributes:
        viewer: The ViewerGL instance to capture from.
        output_dir: Directory where frame buffers will be stored.
        num_cameras: Number of separate recording camera views.
        camera_names: Human-readable names for each recording camera.
    """

    def __init__(
        self,
        viewer: ViewerGL,
        output_dir: str = "./multi_camera_recordings",
        num_cameras: int = 3,
        camera_names: list[str] | None = None,
        skip_frames: int = 0,
        async_save: bool = True,
        depth_model: Model | None = None,
        depth_state_getter: Callable[[], State] | None = None,
    ):
        """Initialize the multi-camera recorder.

        Recording cameras are independent off-screen rendering cameras. The viewer's
        displayed camera is never modified during recording.

        Args:
            viewer: The ViewerGL instance to capture from.
            output_dir: Root directory for storing frame buffers [default: "./multi_camera_recordings"].
            num_cameras: Number of separate camera views [default: 3].
            camera_names: List of names for each camera. If None, uses "camera_0", "camera_1", etc.
            skip_frames: Number of initial frames to skip before recording [default: 0].
            async_save: Whether to save frames asynchronously in background threads [default: True].
            depth_model: Simulation model used to render depth images [default: None].
            depth_state_getter: Callable that returns the current simulation state for
                depth rendering [default: None].
        """
        self.viewer = viewer
        self.output_dir = Path(output_dir)
        self.num_cameras = num_cameras
        self.async_save = async_save
        self.skip_frames = skip_frames
        self._depth_model = depth_model
        self._depth_state_getter = depth_state_getter
        self._depth_enabled = depth_model is not None or depth_state_getter is not None

        if (depth_model is None) != (depth_state_getter is None):
            msg = "depth_model and depth_state_getter must be provided together to enable depth export"
            raise ValueError(msg)

        if camera_names is None:
            self.camera_names = [f"camera_{i}" for i in range(num_cameras)]
        else:
            if len(camera_names) != num_cameras:
                msg = f"camera_names length ({len(camera_names)}) must match num_cameras ({num_cameras})"
                raise ValueError(msg)
            self.camera_names = camera_names

        # Create output directories for each camera
        for name in self.camera_names:
            camera_dir = self.output_dir / name
            camera_dir.mkdir(parents=True, exist_ok=True)
            if self._depth_enabled:
                (camera_dir / "depth").mkdir(parents=True, exist_ok=True)

        # Store camera configurations: (pos, pitch, yaw, fov)
        self._camera_configs: list[dict] = [
            {"pos": wp.vec3(10.0, 0.0, 2.0), "pitch": -5.0, "yaw": -45.0, "fov": 45.0}
            for _ in range(num_cameras)
        ]

        # Frame counter and buffers
        self._frame_idx = 0
        self._frame_buffers: list[wp.array | None] = [None] * num_cameras

        # Cache of temporary camera objects for rendering (one per recording camera)
        self._temp_cameras: list[object | None] = [None] * num_cameras
        self._save_threads: list[threading.Thread] = []
        self._depth_sensor = None
        self._depth_rays: list[wp.array | None] = [None] * num_cameras
        self._depth_buffers: list[wp.array | None] = [None] * num_cameras

    def set_camera_config(
        self, camera_id: int, pos: wp.vec3 | None = None, pitch: float | None = None, yaw: float | None = None, fov: float | None = None
    ) -> None:
        """Configure a camera view.

        Args:
            camera_id: Index of the camera to configure (0 to num_cameras-1).
            pos: Camera position in world coordinates [m].
            pitch: Camera pitch angle in degrees [default: unchanged].
            yaw: Camera yaw angle in degrees [default: unchanged].
            fov: Field of view in degrees [default: unchanged].
        """
        if not 0 <= camera_id < self.num_cameras:
            msg = f"camera_id {camera_id} out of range [0, {self.num_cameras - 1}]"
            raise IndexError(msg)

        if pos is not None:
            self._camera_configs[camera_id]["pos"] = pos
        if pitch is not None:
            self._camera_configs[camera_id]["pitch"] = pitch
        if yaw is not None:
            self._camera_configs[camera_id]["yaw"] = yaw
        if fov is not None:
            self._camera_configs[camera_id]["fov"] = fov

    def get_camera_config(self, camera_id: int) -> dict:
        """Get the configuration of a camera view.

        Args:
            camera_id: Index of the camera to query.

        Returns:
            Dictionary with keys: pos, pitch, yaw, fov.
        """
        if not 0 <= camera_id < self.num_cameras:
            msg = f"camera_id {camera_id} out of range [0, {self.num_cameras - 1}]"
            raise IndexError(msg)
        return self._camera_configs[camera_id].copy()

    def capture_frames(self) -> None:
        """Capture frames from all configured camera views.

        This method should be called during the main simulation loop after calling
        example.render(). It saves the current camera state, switches to each camera
        view, captures the frame, and restores the original camera.

        This operation is efficient as it reuses the viewer's frame buffer and
        optionally saves frames asynchronously without blocking the simulation.
        """
        # Skip frames if requested
        if self._frame_idx < self.skip_frames:
            self._frame_idx += 1
            return

        # Capture frame for each camera using off-screen rendering
        for cam_id in range(self.num_cameras):
            self._capture_single_camera(cam_id)
            if self._depth_enabled:
                self._capture_single_camera_depth(cam_id)

        self._frame_idx += 1

    def _capture_single_camera(self, camera_id: int) -> None:
        """Capture and save a single frame from one recording camera.

        This method renders the scene with a recording camera without affecting
        the displayed view. The recording camera is an independent off-screen
        camera that captures from its own viewpoint.

        Args:
            camera_id: Index of the camera to capture from.
        """
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            msg = "PIL not installed. Frames cannot be saved. Install with: pip install pillow"
            warnings.warn(msg, stacklevel=2)
            return

        # Get or create the recording camera for this view
        config = self._camera_configs[camera_id]
        recording_camera = self._get_or_create_recording_camera(camera_id, config)

        # Render the scene with the recording camera to the frame buffer (off-screen)
        # This does not affect the displayed view
        self.viewer.renderer.render_camera_to_buffer(
            recording_camera,
            self.viewer.objects,
            self.viewer.lines,
        )

        self._ensure_frame_buffer(camera_id)

        # Get frame from viewer's frame buffer as GPU array
        frame = self.viewer.get_frame(target_image=self._frame_buffers[camera_id])

        # Cache buffer for reuse
        if self._frame_buffers[camera_id] is None:
            self._frame_buffers[camera_id] = frame

        # Convert to numpy and PIL
        frame_np = frame.numpy()
        image = Image.fromarray(frame_np, mode="RGB")

        # Generate output path
        camera_dir = self.output_dir / self.camera_names[camera_id]
        filename = camera_dir / f"{self._frame_idx - self.skip_frames:05d}.png"

        # Save asynchronously or synchronously
        if self.async_save:
            save_thread = threading.Thread(target=self._save_image, args=(image, filename), daemon=False)
            self._save_threads.append(save_thread)
            save_thread.start()
        else:
            image.save(str(filename))

    def _get_or_create_recording_camera(self, camera_id: int, config: dict) -> object:
        """Get or create a recording camera object with the given configuration.

        Recording cameras are independent temporary camera objects configured for
        each recording view. They are reused across frames for efficiency.

        Args:
            camera_id: Index of the camera.
            config: Camera configuration dict with keys: pos, pitch, yaw, fov.

        Returns:
            Camera object configured with the given parameters.
        """
        from pyglet.math import Vec3 as PyVec3

        from ._src.viewer.camera import Camera  # noqa: PLC0415

        # Reuse existing camera if available and configuration hasn't changed
        if self._temp_cameras[camera_id] is not None:
            cam = self._temp_cameras[camera_id]
            # Update orientation and FOV if they changed
            pos_tuple = (float(config["pos"][0]), float(config["pos"][1]), float(config["pos"][2]))
            cam.pos = PyVec3(*pos_tuple)
            cam.pitch = max(min(config["pitch"], 89.0), -89.0)
            cam.yaw = (config["yaw"] + 180.0) % 360.0 - 180.0
            cam.fov = config["fov"]
            cam.update_screen_size(*self._get_framebuffer_size())
            return cam

        # Create new camera with same dimensions as viewer camera
        viewer_camera = self.viewer.camera
        framebuffer_width, framebuffer_height = self._get_framebuffer_size()
        pos_tuple = (float(config["pos"][0]), float(config["pos"][1]), float(config["pos"][2]))

        recording_camera = Camera(
            fov=config["fov"],
            near=viewer_camera.near,
            far=viewer_camera.far,
            width=framebuffer_width,
            height=framebuffer_height,
            pos=pos_tuple,
            up_axis=viewer_camera.up_axis,
        )

        recording_camera.pitch = max(min(config["pitch"], 89.0), -89.0)
        recording_camera.yaw = (config["yaw"] + 180.0) % 360.0 - 180.0

        # Cache for reuse
        self._temp_cameras[camera_id] = recording_camera
        return recording_camera

    def _ensure_frame_buffer(self, camera_id: int) -> None:
        """Ensure the cached frame buffer matches the current framebuffer size."""
        height, width = self._get_framebuffer_shape()
        target_shape = (height, width, 3)
        target_buffer = self._frame_buffers[camera_id]

        if target_buffer is not None and target_buffer.shape == target_shape:
            return

        self._frame_buffers[camera_id] = wp.empty(
            shape=target_shape,
            dtype=wp.uint8,  # pyright: ignore[reportArgumentType]
            device=self.viewer.device,
        )

    def _get_framebuffer_shape(self) -> tuple[int, int]:
        """Return the current framebuffer height and width."""
        width, height = self._get_framebuffer_size()
        return height, width

    def _get_framebuffer_size(self) -> tuple[int, int]:
        """Return the current framebuffer width and height."""
        return self.viewer.renderer._screen_width, self.viewer.renderer._screen_height

    def _restore_camera_state(self) -> None:
        """Restore the original camera state (deprecated, kept for compatibility)."""
        # No longer needed - viewer camera is never modified
        pass

    def _save_image(self, image, filename: Path) -> None:
        """Save an image to disk from a background thread."""
        image.save(str(filename))

    def _capture_single_camera_depth(self, camera_id: int) -> None:
        """Capture and save a depth map for one recording camera."""
        self._ensure_depth_sensor()

        state = self._depth_state_getter()
        self._depth_sensor.sync_transforms(state)

        config = self._camera_configs[camera_id]
        camera_transform = wp.array([[self._camera_config_to_transform(config)]], dtype=wp.transform)

        self._ensure_depth_buffer(camera_id)
        camera_rays = self._get_depth_rays(camera_id)

        self._depth_sensor.update(
            state,
            camera_transform,
            camera_rays,
            color_image=None,
            depth_image=self._depth_buffers[camera_id],
        )

        depth_np = self._depth_buffers[camera_id].numpy()[0, 0]
        depth_path = self.output_dir / self.camera_names[camera_id] / "depth" / f"{self._frame_idx - self.skip_frames:05d}.npy"
        self._save_depth_array(depth_np, depth_path)

    def _ensure_depth_sensor(self) -> None:
        """Create the tiled camera sensor lazily when depth export is enabled."""
        if self._depth_sensor is not None:
            return

        from .sensors import SensorTiledCamera  # noqa: PLC0415

        self._depth_sensor = SensorTiledCamera(
            model=self._depth_model,
            config=SensorTiledCamera.Config(
                default_light=True,
                default_light_shadows=True,
                checkerboard_texture=True,
                backface_culling=True,
            ),
        )

    def _ensure_depth_buffer(self, camera_id: int) -> None:
        """Ensure the cached depth buffer matches the current framebuffer size."""
        height, width = self._get_framebuffer_shape()
        target_shape = (1, 1, height, width)
        target_buffer = self._depth_buffers[camera_id]

        if target_buffer is not None and target_buffer.shape == target_shape:
            return

        self._depth_buffers[camera_id] = self._depth_sensor.create_depth_image_output(width, height, 1)
        self._depth_rays[camera_id] = None

    def _get_depth_rays(self, camera_id: int) -> wp.array:
        """Return cached pinhole rays for the current camera configuration."""
        if self._depth_rays[camera_id] is not None:
            return self._depth_rays[camera_id]

        import math  # noqa: PLC0415

        width, height = self._get_framebuffer_size()
        fov_radians = math.radians(self._camera_configs[camera_id]["fov"])
        self._depth_rays[camera_id] = self._depth_sensor.compute_pinhole_camera_rays(width, height, fov_radians)
        return self._depth_rays[camera_id]

    def _camera_config_to_transform(self, config: dict) -> wp.transform:
        """Convert a recorder camera config into a SensorTiledCamera transform."""
        import math  # noqa: PLC0415

        pitch = math.radians(config["pitch"])
        yaw = math.radians(config["yaw"])

        quat_pitch = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), pitch)
        quat_yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)

        return wp.transform(config["pos"], quat_yaw * quat_pitch)

    def _save_depth_array(self, depth: object, filename: Path) -> None:
        """Save a depth array to disk."""
        import numpy as np  # noqa: PLC0415

        np.save(str(filename), depth)

    def _wait_for_pending_saves(self) -> None:
        """Wait for all background image saves to finish."""
        if not self._save_threads:
            return

        pending_threads = self._save_threads
        self._save_threads = []

        for save_thread in pending_threads:
            save_thread.join()

    def generate_videos(
        self,
        fps: int = 30,
        codec: str = "libx264",
        quality: int = 5,
        keep_frames: bool = False,
    ) -> dict[str, bool]:
        """Generate MP4 videos from captured frames for all cameras.

        Uses imageio-ffmpeg for H.264 video encoding. Each camera gets its own
        video file. Requires imageio-ffmpeg to be installed.

        Args:
            fps: Frames per second for output videos [default: 30].
            codec: FFmpeg codec to use [default: "libx264"].
            quality: Quality level (1-10, lower is better quality) [default: 5].
            keep_frames: Whether to keep PNG frames after video generation [default: False].

        Returns:
            Dictionary mapping camera names to success status (True if video generated,
            False if frames were empty or encoding failed).

        Example:
            >>> results = recorder.generate_videos(fps=60, keep_frames=False)
            >>> for camera_name, success in results.items():
            ...     if success:
            ...         print(f"Generated {camera_name}.mp4")
        """
        try:
            import imageio_ffmpeg as ffmpeg  # noqa: PLC0415
        except ImportError:
            msg = "imageio-ffmpeg not installed. Videos cannot be generated. Install with: pip install imageio-ffmpeg"
            warnings.warn(msg, stacklevel=2)
            return {name: False for name in self.camera_names}
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            msg = "PIL not installed. Videos cannot be generated. Install with: pip install pillow"
            warnings.warn(msg, stacklevel=2)
            return {name: False for name in self.camera_names}

        import numpy as np  # noqa: PLC0415

        self._wait_for_pending_saves()

        results = {}

        # Generate video for each camera
        for cam_id, camera_name in enumerate(self.camera_names):
            camera_dir = self.output_dir / camera_name
            frame_files = sorted(glob.glob(str(camera_dir / "*.png")))

            if not frame_files:
                msg = f"No PNG frames found in {camera_dir}"
                warnings.warn(msg, stacklevel=2)
                results[camera_name] = False
                continue

            output_filename = self.output_dir / f"{camera_name}.mp4"

            try:
                # Read first frame to get dimensions; ensure even size for H.264 encoding.
                with Image.open(frame_files[0]) as first_frame:
                    width, height = first_frame.size

                even_width = width if width % 2 == 0 else width + 1
                even_height = height if height % 2 == 0 else height + 1
                needs_padding = even_width != width or even_height != height

                # Use imageio-ffmpeg to write video
                writer = ffmpeg.write_frames(
                    str(output_filename),
                    size=(even_width, even_height),
                    fps=fps,
                    codec=codec,
                    macro_block_size=1,
                    quality=quality,
                )
                writer.send(None)  # Initialize

                # Send each frame to the encoder
                for frame_path in frame_files:
                    with Image.open(frame_path) as frame_image:
                        frame_array = np.array(frame_image.convert("RGB"))

                    if needs_padding:
                        padded_frame = np.zeros((even_height, even_width, frame_array.shape[2]), dtype=frame_array.dtype)
                        padded_frame[:height, :width] = frame_array
                        frame_array = padded_frame

                    writer.send(frame_array)

                writer.close()

                msg = f"Generated {output_filename} from {len(frame_files)} frames at {fps} fps"
                warnings.warn(msg, category=UserWarning, stacklevel=2)

                results[camera_name] = True

                # Clean up frames if requested
                if not keep_frames:
                    for frame_path in frame_files:
                        os.remove(frame_path)

            except Exception as e:
                msg = f"Failed to generate video for {camera_name}: {e}"
                warnings.warn(msg, stacklevel=2)
                results[camera_name] = False

        return results

    def get_frame_count(self) -> int:
        """Get the number of frames that have been captured so far.

        Returns:
            Number of capture calls made (after skip_frames).
        """
        return max(0, self._frame_idx - self.skip_frames)

    def reset(self) -> None:
        """Reset the recorder state.

        Clears frame counter and camera cache. Useful if you want to record
        multiple simulations with the same recorder instance.
        """
        self._frame_idx = 0
        self._temp_cameras = [None] * self.num_cameras
        self._depth_rays = [None] * self.num_cameras
        self._depth_buffers = [None] * self.num_cameras
        self._wait_for_pending_saves()
        self._save_threads = []
