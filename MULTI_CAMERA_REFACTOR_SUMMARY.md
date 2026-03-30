# Multi-Camera Recorder Refactoring - Solution Summary

## Problem Statement
The original multi-camera recorder implementation would switch the viewer's display camera to each recording camera and call `end_frame()` to render. This caused the on-screen camera to jump between different viewpoints during simulation, creating a jarring viewing experience.

## Solution Overview
Refactored the multi-camera recorder to use independent off-screen recording cameras that **never affect the displayed simulation view**. The visible camera remains fixed while 3+ recording cameras capture frames independently.

## Key Changes

### 1. New RendererGL Method: `render_camera_to_buffer()`
**File**: `newton/_src/viewer/gl/opengl.py`

Added a new public method to the RendererGL class that enables off-screen rendering with a custom camera:

```python
def render_camera_to_buffer(self, camera, objects, lines=None):
    """Render the scene with a custom camera to the internal frame buffer.
    
    This method renders using a custom camera without modifying the main camera state.
    Useful for off-screen rendering and multi-camera capture.
    """
```

**Why**: Allows rendering to the frame buffer with different cameras without switching the viewer's main camera.

**Implementation Details**:
- Takes a custom camera object as parameter
- Renders to the internal frame buffer (no screen output)
- Temporarily replaces `self.camera` during rendering only
- Restores original camera state after rendering
- Handles shadow mapping, MSAA resolution, and all normal rendering pipeline steps
- Compatible with existing viewport and framebuffer setup

### 2. MultiCameraRecorder Refactoring
**File**: `newton/multi_camera_recorder.py`

#### A. Removed Camera State Tracking
- **Deleted**: `_original_pos`, `_original_pitch`, `_original_yaw`, `_original_fov`
- **Reason**: No longer needed since we never modify the viewer's camera

#### B. Added Recording Camera Cache
- **Added**: `_temp_cameras` - list of independent Camera objects for each recording view
- **Benefit**: Cameras are created once and reused across frames for efficiency

#### C. Refactored `capture_frames()`
**Before**:
```python
# Save original camera state
# For each camera:
#   - Call set_camera() to switch viewer camera
#   - Call end_frame() to render with new camera
#   - Call get_frame() to capture
# Restore original camera
```

**After**:
```python
# For each camera:
#   - Get or create recording camera
#   - Call renderer.render_camera_to_buffer() (off-screen)
#   - Call get_frame() to capture
# Done! No camera state to manage
```

#### D. New Helper Method: `_get_or_create_recording_camera()`
Creates and manages independent Camera objects for each recording view:

```python
def _get_or_create_recording_camera(self, camera_id: int, config: dict) -> object:
    """Get or create a recording camera object with the given configuration."""
```

**Features**:
- Reuses existing camera objects across frames
- Updates camera position, orientation, and FOV based on config
- Maintains same dimensions as viewer camera
- Respects camera up-axis convention

#### E. Updated `_capture_single_camera()`
Now uses off-screen rendering:

```python
recording_camera = self._get_or_create_recording_camera(camera_id, config)

# Off-screen render - doesn't affect displayed view
self.viewer.renderer.render_camera_to_buffer(
    recording_camera,
    self.viewer.objects,
    self.viewer.lines,
)

# Read the rendered framebuffer
frame = self.viewer.get_frame(target_image=self._frame_buffers[camera_id])
```

#### F. Simplified `reset()`
Now only clears frame counter and camera cache:

```python
def reset(self) -> None:
    self._frame_idx = 0
    self._temp_cameras = [None] * self.num_cameras
```

### 3. Updated Documentation
**File**: `newton/examples/basic/example_multi_camera_recording.py`

Updated example docstring to emphasize the key improvement:
- Recording cameras are now independent off-screen cameras
- The visible simulation window is never affected by recording operations
- No on-screen camera jumping during recording

## API Compatibility
✅ **Fully backward compatible** - The public API remains unchanged:
- `MultiCameraRecorder(viewer, ...)`
- `set_camera_config(camera_id, pos, pitch, yaw, fov)`
- `capture_frames()`
- `generate_videos(fps, codec, quality, keep_frames)`
- `get_frame_count()`
- `reset()`

Users don't need to change any code!

## Benefits

### For Users
1. **No Camera Jumping** - The displayed view stays fixed during recording
2. **Independent Recording** - Each recording camera operates independently
3. **Better UX** - Can watch the simulation normally while recording happens off-screen
4. **Same API** - Existing code works without changes

### For Performance
1. **Efficient Rendering** - Off-screen rendering doesn't affect display pipeline
2. **Camera Reuse** - Recording cameras are cached and reused across frames
3. **No Extra Calls** - Eliminates redundant `set_camera()` and `end_frame()` calls
4. **Minimal Overhead** - Only adds one `render_camera_to_buffer()` call per recording camera

### For Architecture
1. **Cleaner Separation** - Recording cameras (off-screen) separate from display camera (on-screen)
2. **Extensible** - RendererGL `render_camera_to_buffer()` can be used for other purposes
3. **Maintainable** - No camera state management complexity in MultiCameraRecorder

## Testing
All refactoring validated through:
1. ✓ RendererGL has `render_camera_to_buffer()` method
2. ✓ MultiCameraRecorder initializes correctly
3. ✓ Camera objects can be created and configured
4. ✓ Old camera state variables removed
5. ✓ capture_frames() no longer switches viewer camera
6. ✓ Off-screen rendering method is called
7. ✓ Example documentation updated

## Migration Guide
No migration needed! The refactoring is transparent to users:

```python
# Your existing code works as-is:
recorder = multi_camera_recorder.MultiCameraRecorder(viewer)
recorder.set_camera_config(0, pos=wp.vec3(10, 0, 2), ...)
recorder.set_camera_config(1, pos=wp.vec3(-10, 0, 2), ...)
recorder.set_camera_config(2, pos=wp.vec3(0, 10, 2), ...)

for frame in range(num_frames):
    example.step()
    example.render()
    recorder.capture_frames()  # Now uses off-screen rendering!

recorder.generate_videos(fps=30)
```

The only difference: the displayed camera no longer jumps between recording views!

## Future Enhancements
The `render_camera_to_buffer()` method in RendererGL enables:
1. Custom viewport rendering to any framebuffer
2. Multi-pass rendering techniques
3. Render-to-texture workflows
4. Advanced camera effects without affecting display
5. Post-processing with different camera parameters

## Files Modified
1. `newton/_src/viewer/gl/opengl.py` - Added `render_camera_to_buffer()` method (94 lines)
2. `newton/multi_camera_recorder.py` - Refactored to use off-screen rendering (~150 lines changed)
3. `newton/examples/basic/example_multi_camera_recording.py` - Updated documentation
