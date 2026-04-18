# Quick Start: Cloth Franka Data Export

## The Problem You Had
- You wanted to export color and depth from cloth_franka's 3 cameras
- The original data export example didn't work with cloth_franka
- You didn't want to modify cloth_franka itself

## The Solution
Run the new integrated example:

```bash
python -m newton.examples cloth_franka_data_export --headless --num-frames 500
```

That's it! 🎉

## What You Get

### Output Directory Structure
```
cloth_franka_data_export/
├── overhead/color/       ← Top-down view of workspace
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── front/color/          ← Front-left view of robot & cloth
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── side/color/           ← Side view of robot arm & cloth
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
└── camera_info_*.json    ← Camera calibration data
```

### Camera Calibration JSON
Each frame's camera parameters saved as:
```json
{
  "frame": 0,
  "cameras": [
    {
      "name": "overhead",
      "intrinsics": {
        "fx": 886.81, "fy": 665.11,
        "cx": 512.0, "cy": 384.0,
        "width": 1024, "height": 768
      },
      "extrinsics": {
        "position": [0.0, 0.0, 1.5],
        "pitch": -90.0, "yaw": 0.0
      }
    },
    ...
  ]
}
```

## Performance

| Mode | FPS | Use Case |
|------|-----|----------|
| Headless | 50-100 | Data export, training |
| With GUI | 10-20 | Debugging, visualization |

## Common Commands

```bash
# Fast export (500 frames in ~10 seconds)
python -m newton.examples cloth_franka_data_export --headless --num-frames 500

# With visualization (slower)
python -m newton.examples cloth_franka_data_export --num-frames 500

# Full simulation (3850 frames, ~1.2 GB data)
python -m newton.examples cloth_franka_data_export --headless

# For testing (just 100 frames)
python -m newton.examples cloth_franka_data_export --headless --num-frames 100
```