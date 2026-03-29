# Multi-Camera Recording Guide

## Overview

The multi-camera recording feature in Newton allows you to record a simulation from multiple camera viewpoints simultaneously and export them as separate video files. This is useful for:

- Analyzing robot behavior from different perspectives
- Creating documentation videos with multiple viewpoints
- Debugging complex dynamics that are difficult to understand from a single angle
- Generating training data for computer vision applications

The feature integrates seamlessly into the existing simulation loop without requiring any changes to the core architecture.

## Quick Start

### 1. Import the Module

```python
import warp as wp
from newton import viewer, multi_camera_recorder
```

### 2. Create a Viewer and Recorder

```python
# Create OpenGL viewer
v = viewer.ViewerGL()

# Create multi-camera recorder with default 3 cameras
recorder = multi_camera_recorder.MultiCameraRecorder(
    v,
    output_dir="./recordings",
    num_cameras=3,
)
```

### 3. Configure Camera Views

```python
# Configure camera 0: Front view
recorder.set_camera_config(
    0,
    pos=wp.vec3(10.0, 0.0, 2.0),
    pitch=-5.0,
    yaw=-45.0,
    fov=45.0,
)

# Configure camera 1: Side view
recorder.set_camera_config(
    1,
    pos=wp.vec3(0.0, -15.0, 2.0),
    pitch=-10.0,
    yaw=-90.0,
    fov=45.0,
)

# Configure camera 2: Top view
recorder.set_camera_config(
    2,
    pos=wp.vec3(0.0, 0.0, 12.0),
    pitch=-90.0,
    yaw=0.0,
    fov=60.0,
)
```

### 4. Capture Frames During Simulation

In your main simulation loop, call `capture_frames()` after rendering:

```python
while viewer.is_running():
    if not viewer.is_paused():
        example.step()           # Advance simulation
    example.render()             # Render frame
    recorder.capture_frames()    # Capture from all cameras
```

### 5. Generate Videos

After the simulation completes, generate the video files:

```python
results = recorder.generate_videos(fps=30, keep_frames=False)

for camera_name, success in results.items():
    if success:
        print(f"Generated {camera_name}.mp4")
```

## Configuration Options

### MultiCameraRecorder Constructor

```python
recorder = multi_camera_recorder.MultiCameraRecorder(
    viewer,                    # ViewerGL instance
    output_dir="./recordings",  # Root directory for frame storage
    num_cameras=3,             # Number of camera views
    camera_names=None,         # List[str], defaults to "camera_0", etc.
    skip_frames=0,             # Skip first N frames before recording
    async_save=True,           # Save frames asynchronously
)
```

**Parameters:**

- `viewer`: The `ViewerGL` instance to capture from
- `output_dir`: Directory where output frames and videos will be stored. Each camera gets its own subdirectory.
- `num_cameras`: Number of separate camera views (default: 3)
- `camera_names`: Human-readable names for each camera. If `None`, defaults to `"camera_0"`, `"camera_1"`, etc.
- `skip_frames`: Skip the first N frames before recording (useful to let simulation settle, default: 0)
- `async_save`: Save frames asynchronously in background threads without blocking simulation (default: True)

### Camera Configuration

```python
recorder.set_camera_config(
    camera_id,          # int: 0 to num_cameras-1
    pos=None,           # wp.vec3: Camera position [m]
    pitch=None,         # float: Pitch angle [degrees]
    yaw=None,           # float: Yaw angle [degrees]
    fov=None,           # float: Field of view [degrees]
)
```

All parameters are optional. Only the parameters you specify will be updated.

### Generate Videos

```python
results = recorder.generate_videos(
    fps=30,            # Frames per second (default: 30)
    codec="libx264",   # FFmpeg codec (default: "libx264" - H.264)
    quality=5,         # Quality level 1-10; lower is better (default: 5)
    keep_frames=False,  # Keep PNG frames after encoding (default: False)
)
```

**Returns:** Dictionary mapping camera names to success status (True/False)

## Camera Positioning Guide

When configuring your camera views, consider these standard positions:

### Front Isometric View
```python
recorder.set_camera_config(0,
    pos=wp.vec3(12.0, -8.0, 3.0),
    pitch=-15.0,
    yaw=-135.0,
    fov=45.0,
)
```

### Side View (90° perpendicular)
```python
recorder.set_camera_config(1,
    pos=wp.vec3(0.0, -15.0, 3.0),
    pitch=-10.0,
    yaw=-90.0,
    fov=45.0,
)
```

### Top View (Overhead)
```python
recorder.set_camera_config(2,
    pos=wp.vec3(0.0, 0.0, 12.0),
    pitch=-90.0,  # Looking straight down
    yaw=0.0,
    fov=60.0,
)
```

### Back View
```python
recorder.set_camera_config(3,
    pos=wp.vec3(-12.0, 0.0, 3.0),
    pitch=-5.0,
    yaw=45.0,
    fov=45.0,
)
```

## Advanced Usage

### Dynamic Camera Names

```python
recorder = multi_camera_recorder.MultiCameraRecorder(
    v,
    num_cameras=4,
    camera_names=["top_down", "front_left", "back_right", "side"],
)
```

### Querying Camera Configuration

```python
# Get the configuration of a camera
config = recorder.get_camera_config(camera_id=0)
print(f"Camera 0 position: {config['pos']}")
print(f"Camera 0 FOV: {config['fov']}")
```

### Monitoring Frame Capture

```python
# Get the number of frames captured (after skip_frames)
frame_count = recorder.get_frame_count()
print(f"Captured {frame_count} frames")
```

### Resetting the Recorder

```python
# If running multiple simulations with the same recorder
recorder.reset()
```

### Custom Frame Skip

Skip the first 30 frames to let the simulation settle before recording:

```python
recorder = multi_camera_recorder.MultiCameraRecorder(
    v,
    skip_frames=30,  # Skip first 30 frames
)
```

### Synchronous Frame Saving

For better synchronization with the simulation loop (may impact performance):

```python
recorder = multi_camera_recorder.MultiCameraRecorder(
    v,
    async_save=False,  # Block until each frame is saved
)
```

## Output Structure

After running the simulation and generating videos, you'll have:

```
./recordings/
├── camera_0/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── camera_1/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── camera_2/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── camera_0.mp4
├── camera_1.mp4
└── camera_2.mp4
```

## Requirements

The multi-camera recorder requires:

- **For frame capture**: `Pillow` (PIL)
  ```bash
  pip install pillow
  ```

- **For video generation**: `imageio-ffmpeg`
  ```bash
  pip install imageio-ffmpeg
  ```

If either dependency is missing, the recorder will issue warnings and skip that functionality.

## Performance Considerations

### Asynchronous Saving (Recommended)
With `async_save=True` (default), frame saving happens in background threads:
- ✓ Minimal impact on simulation performance
- ✓ Simulation loop proceeds immediately after `capture_frames()`
- ⚠ Files are written in the background; ensure proper cleanup

### Synchronous Saving
With `async_save=False`:
- ✓ Deterministic ordering of file writes
- ✓ Easier debugging if issues occur
- ⚠ Simulation loop blocks while saving frames (impacts FPS)

### Frame Skip Strategy
Use `skip_frames` to skip initial frames before recording:
```python
# Skip first 60 frames (1 second at 60fps) to let simulation settle
recorder = multi_camera_recorder.MultiCameraRecorder(v, skip_frames=60)
```

This reduces disk I/O and focuses recordings on interesting simulation behavior.

## Example: Complete Integration

```python
import warp as wp
import newton
from newton import viewer, multi_camera_recorder

class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        # ... setup simulation ...
        
        # Initialize multi-camera recorder
        self.recorder = multi_camera_recorder.MultiCameraRecorder(
            viewer,
            output_dir="./sim_recordings",
            num_cameras=3,
            skip_frames=30,
        )
        
        # Configure standard views
        self.recorder.set_camera_config(0, pos=wp.vec3(10, -8, 3), 
                                        pitch=-15, yaw=-135)
        self.recorder.set_camera_config(1, pos=wp.vec3(0, -15, 3), 
                                        pitch=-10, yaw=-90)
        self.recorder.set_camera_config(2, pos=wp.vec3(0, 0, 12), 
                                        pitch=-90, yaw=0)
    
    def step(self):
        # Advance simulation
        self.solver.step(self.state_0, self.state_1)
    
    def render(self):
        # Render frame
        self.viewer.begin_frame(self.sim_time)
        self.viewer.set_model_state(self.state_0)
        self.viewer.end_frame()
        
        # Capture from all cameras
        self.recorder.capture_frames()
    
    def test_final(self):
        # Generate videos after simulation
        print(f"Recorded {self.recorder.get_frame_count()} frames")
        results = self.recorder.generate_videos(fps=60, keep_frames=False)
        
        for name, success in results.items():
            print(f"{'✓' if success else '✗'} {name}.mp4")
```

## Troubleshooting

### "PIL not installed" Warning
```bash
pip install pillow
```

### "imageio-ffmpeg not installed" Warning
```bash
pip install imageio-ffmpeg
```

### No frames captured
- Ensure you're calling `recorder.capture_frames()` after `example.render()` in the main loop
- Check that `skip_frames` is not larger than the total simulation frames
- Verify the output directory is writable

### Video generation fails
- Check that `imageio-ffmpeg` is properly installed
- Ensure all PNG frames were successfully saved (check output directory)
- Verify sufficient disk space and memory for video encoding

### Viewer glitches during multi-camera capture
- The recorder temporarily switches camera views during rendering
- This is expected and should not affect the output video quality
- The simulation remains synchronized with the UI view
