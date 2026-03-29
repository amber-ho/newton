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
and export them as separate video files. It integrates seamlessly with the existing
simulation loop without requiring architectural changes.

Example:
    >>> from newton import viewer, multi_camera_recorder
    >>> import warp as wp
    >>>
    >>> # Create viewer and recorder
    >>> v = viewer.ViewerGL()
    >>> recorder = multi_camera_recorder.MultiCameraRecorder(v, output_dir="./recordings")
    >>>
    >>> # Configure camera views
    >>> recorder.set_camera_config(0, pos=wp.vec3(10, 0, 2), pitch=-5, yaw=-45)
    >>> recorder.set_camera_config(1, pos=wp.vec3(-10, 0, 2), pitch=-5, yaw=135)
    >>> recorder.set_camera_config(2, pos=wp.vec3(0, 10, 2), pitch=-5, yaw=-135)
    >>>
    >>> # During simulation loop, capture frames
    >>> for frame in range(num_frames):
    ...     example.step()
    ...     example.render()
    ...     recorder.capture_frames()  # Captures all 3 camera views
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
    from ._src.viewer import ViewerGL

__all__ = ["MultiCameraRecorder"]


class MultiCameraRecorder:
    """Records simulation frames from multiple camera viewpoints.

    This class manages capture of simulation frames from multiple camera configurations,
    storing them separately for later video generation. It hooks into the existing
    simulation loop without modifying the viewer's core architecture.

    Attributes:
        viewer: The ViewerGL instance to capture from.
        output_dir: Directory where frame buffers will be stored.
        num_cameras: Number of separate camera views to record.
        camera_names: Human-readable names for each camera.
    """

    def __init__(
        self,
        viewer: ViewerGL,
        output_dir: str = "./multi_camera_recordings",
        num_cameras: int = 3,
        camera_names: list[str] | None = None,
        skip_frames: int = 0,
        async_save: bool = True,
    ):
        """Initialize the multi-camera recorder.

        Args:
            viewer: The ViewerGL instance to capture from.
            output_dir: Root directory for storing frame buffers [default: "./multi_camera_recordings"].
            num_cameras: Number of separate camera views [default: 3].
            camera_names: List of names for each camera. If None, uses "camera_0", "camera_1", etc.
            skip_frames: Number of initial frames to skip before recording [default: 0].
            async_save: Whether to save frames asynchronously in background threads [default: True].
        """
        self.viewer = viewer
        self.output_dir = Path(output_dir)
        self.num_cameras = num_cameras
        self.async_save = async_save
        self.skip_frames = skip_frames

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

        # Store camera configurations: (pos, pitch, yaw, fov)
        self._camera_configs: list[dict] = [
            {"pos": wp.vec3(10.0, 0.0, 2.0), "pitch": -5.0, "yaw": -45.0, "fov": 45.0}
            for _ in range(num_cameras)
        ]

        # Frame counter and buffers
        self._frame_idx = 0
        self._frame_buffers: list[wp.array | None] = [None] * num_cameras

        # Store original camera state for restoration
        self._original_pos = None
        self._original_pitch = None
        self._original_yaw = None
        self._original_fov = None

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

        # Save original camera state on first capture
        if self._original_pos is None:
            self._original_pos = self.viewer.camera.pos.copy()
            self._original_pitch = self.viewer.camera.pitch
            self._original_yaw = self.viewer.camera.yaw
            self._original_fov = self.viewer.camera.fov

        # Capture frame for each camera
        for cam_id in range(self.num_cameras):
            self._capture_single_camera(cam_id)

        # Restore original camera state
        self._restore_camera_state()

        self._frame_idx += 1

    def _capture_single_camera(self, camera_id: int) -> None:
        """Capture and save a single frame from one camera view.

        Args:
            camera_id: Index of the camera to capture from.
        """
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            msg = "PIL not installed. Frames cannot be saved. Install with: pip install pillow"
            warnings.warn(msg, stacklevel=2)
            return

        # Get camera configuration
        config = self._camera_configs[camera_id]

        # Switch to this camera view
        self.viewer.set_camera(
            pos=config["pos"],
            pitch=config["pitch"],
            yaw=config["yaw"],
        )
        self.viewer.camera.fov = config["fov"]

        # Re-render with this camera (end_frame will use new camera)
        self.viewer.end_frame()

        # Get frame from viewer as GPU array
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
            threading.Thread(target=image.save, args=(str(filename),), daemon=False).start()
        else:
            image.save(str(filename))

    def _restore_camera_state(self) -> None:
        """Restore the original camera state."""
        if self._original_pos is not None:
            self.viewer.set_camera(
                pos=self._original_pos,
                pitch=self._original_pitch,
                yaw=self._original_yaw,
            )
            self.viewer.camera.fov = self._original_fov

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
                # Get frame dimensions from first frame
                from PIL import Image  # noqa: PLC0415

                first_frame = Image.open(frame_files[0])
                width, height = first_frame.size

                # Use imageio-ffmpeg to write video
                writer = ffmpeg.write_frames(
                    str(output_filename),
                    size=(width, height),
                    fps=fps,
                    codec=codec,
                    macro_block_size=8,
                    quality=quality,
                )
                writer.send(None)  # Initialize

                # Send each frame to the encoder
                for frame_path in frame_files:
                    frame_image = Image.open(frame_path)
                    import numpy as np  # noqa: PLC0415

                    frame_array = np.array(frame_image)
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

        Clears frame counter and camera state. Useful if you want to record
        multiple simulations with the same recorder instance.
        """
        self._frame_idx = 0
        self._original_pos = None
        self._original_pitch = None
        self._original_yaw = None
        self._original_fov = None
