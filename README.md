# Newton

## PhysTwin physical property integration
`newton/load_phystwin_npz.py`: This is a file that will convert `physics_param.npz` into the data format that Newton can use.
Not needed.
***
How to use:
```bash
uv run -m newton.examples cloth_phystwin --npz-path physics_export/physics_params.npz
```