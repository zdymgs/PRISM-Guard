# Competition environment

From the repository root, install the competition dependencies with:

```bash
python -m pip install -e ".[competition]"
```

Use Python 3.12 and a PyTorch build compatible with your hardware. The frozen challenge environment used PyTorch 2.9.1 and torchvision 0.24.1; [requirements.txt](requirements.txt) lists the direct supporting packages and [requirements-lock.txt](requirements-lock.txt) records the tested Windows CUDA 13.0 environment.

The CUDA-specific lock is an environment record. Select the appropriate PyTorch wheel for your platform from [the PyTorch installation guide](https://pytorch.org/get-started/locally/), then install the project extra above. Inference uses the supplied local deployment bundle and raw input directory. Data, model weights, virtual environments, and downloaded packages are not included.
