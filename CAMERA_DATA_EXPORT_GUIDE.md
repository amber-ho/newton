# Camera Data Export Guide

## Using the Components

### Option A: With Cloth Franka (Current Approach)

Use the integrated example as shown above. `cloth_franka` now enables depth
export through `MultiCameraRecorder`, so each camera writes:

- RGB frames as `./cloth_franka_recordings/<camera>/<frame>.png`
- Depth frames as `./cloth_franka_recordings/<camera>/depth/<frame>.npy`

### Option B: With Your Own Simulation

To use `CameraDataExporter` with a different Newton example:

```python
from newton.examples.data_export.camera_data_exporter import CameraDataExporter
import warp as wp

# Setup your simulation
your_model = build_your_model()
your_state = your_model.state()

# Create exporter with custom cameras
exporter = CameraDataExporter(
    model=your_model,
    state=your_state,
    camera_configs=[
        {
            "pos": wp.vec3(0.0, 0.0, 2.0),
            "pitch": -90.0,
            "yaw": 0.0,
            "fov": 60.0,
        },
        {
            "pos": wp.vec3(3.0, 0.0, 1.0),
            "pitch": -30.0,
            "yaw": 0.0,
            "fov": 45.0,
        },
    ],
    camera_names=["top", "side"],
    output_dir="./my_export",
    render_width=640,
    render_height=480,
)

# In your simulation loop:
for frame in range(num_frames):
    your_model.step()
    exporter.render_frame(your_state, frame)
    exporter.export_camera_info(frame)
```

## Camera Configuration

Each camera in `camera_configs` is defined by:

- **pos** (wp.vec3): Camera position in world coordinates [m]
- **pitch** (float): Pitch angle in degrees (positive = looking up, negative = looking down)
- **yaw** (float): Yaw angle in degrees (rotation around Z axis)
- **fov** (float): Field of view in degrees

### Cloth Franka Cameras

The default cloth_franka cameras are configured as:

| Camera | Position | Pitch | Yaw | FOV | Purpose |
|--------|----------|-------|-----|-----|---------|
| **Overhead** | (0, 0, 1.5) | -90° | 0° | 60° | Looking straight down at workspace |
| **Front** | (1.0, 0.8, 0.5) | -30° | -135° | 50° | Front-left view of robot & cloth |
| **Side** | (0, 1.2, 0.7) | -25° | -90° | 50° | Side view of robot arm & cloth |

## Depth Export With MultiCameraRecorder

`MultiCameraRecorder` can now export depth alongside RGB when you pass both:

- `depth_model`: the simulation model
- `depth_state_getter`: a callable returning the current simulation state

Example:

```python
recorder = multi_camera_recorder.MultiCameraRecorder(
    viewer,
    output_dir="./recordings",
    num_cameras=2,
    camera_names=["top", "side"],
    depth_model=model,
    depth_state_getter=lambda: state,
)
```

Depth files are saved as `./recordings/<camera>/depth/<frame>.npy`.

## Limitations & Future Work

### Current Limitations

1. **Depth Needs Simulation State**
   - Depth export depends on `SensorTiledCamera`, so the recorder needs access to
     the live simulation state through `depth_state_getter`

2. **Sensor and Viewer Rendering Can Differ Slightly**
   - RGB frames come from the viewer renderer
   - Depth frames come from `SensorTiledCamera`
   - Small visual differences are possible if the two render paths diverge

### Low Frame Rate

If running slow:

1. Use `--headless` flag (much faster)
2. Reduce `--num-frames` for testing
3. Reduce render resolution in `CameraDataExporter` (change 640x480 → 320x240)
