# ONNX to RKNN

This is script to convert PP-YOLOE model from ONNX format to RKNN format.
Must be run on hardware with Rockchip chips rk3588 rk3576.

Install dependencies:

```bash
uv sync
```

Run script:

```bash
uv run main.py --dataset {dataset.txt path} --model {onnx model path} --platform {rk3588/rk3576} --width {width} --height {height}
```

Options:

| Option | Default | Description |
| --- | --- | --- |
| `--model` | | Path to the ONNX model file |
| `--platform` | `rk3588` | Target platform (`rk3588` or `rk3576`) |
| `--dataset` | | Path to dataset txt file for quantization (optional) |
| `--width` | `640` | Input image width |
| `--height` | `640` | Input image height |
