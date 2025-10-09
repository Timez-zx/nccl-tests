## Working environment and runbook (validated on this host)

This document captures the exact environment and commands that successfully build and run `nccl-tests` on this machine.

### System
- GPU: NVIDIA A100 80GB PCIe
- NVIDIA driver: 525.125.06
- CUDA driver API reported by `nvidia-smi`: CUDA Version 12.0

Quick checks:
```bash
nvidia-smi
```

### Toolchain and libraries
- CUDA Toolkit used to build: 12.0 (nvcc 12.0.140) at `/usr/local/cuda-12.0`
  - Check: `\/usr\/local\/cuda-12.0\/bin\/nvcc --version`
- NCCL packages (pinned):
  - `libnccl2=2.18.5-1+cuda12.0`
  - `libnccl-dev=2.18.5-1+cuda12.0`
  - Held to prevent automatic upgrades.
- `nccl-tests` source tag: `v2.17.2`

List installed NCCL versions:
```bash
apt-cache policy libnccl2 libnccl-dev | sed -n '1,80p'
```

### One-time setup (NCCL 2.18.5 for CUDA 12.0)

Install and hold matching NCCL packages (requires sudo):
```bash
sudo apt-get update
sudo apt-get install -y --allow-downgrades \
  'libnccl2=2.18.5-1+cuda12.0' \
  'libnccl-dev=2.18.5-1+cuda12.0'
sudo apt-mark hold libnccl2 libnccl-dev
```

Optional: verify the CUDA toolkit version used for builds:
```bash
/usr/local/cuda-12.0/bin/nvcc --version
```

### Build (no environment exports; pass variables to make)

From the repository root `nccl-tests`:
```bash
make clean
CUDA_HOME=/usr/local/cuda-12.0 \
NVCC=/usr/local/cuda-12.0/bin/nvcc \
NVCC_GENCODE='-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80' \
make -j"$(nproc)" MPI=0 VERBOSE=1
```

Notes:
- Target architecture is SM80 only (A100).
- Using make variables avoids polluting the shell environment.

### Run

Run with the CUDA 12.0 runtime on library path; enable NCCL debug for visibility:
```bash
cd build
LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH \
NCCL_DEBUG=INFO \
./all_reduce_perf -b 8M -e 512M -f 2
```

You should see output similar to:
```text
NCCL version 2.18.5+cuda12.0
# size ... time ... algbw ... #wrong 0
```

Confirm the binary links against the expected CUDA and NCCL at runtime:
```bash
LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH \
ldd ./all_reduce_perf | sed -n '1,120p'
# Expect libcudart.so.12 => /usr/local/cuda-12.0/lib64/libcudart.so.12
# and libnccl.so.2 => /lib/x86_64-linux-gnu/libnccl.so.2 (2.18.5-1+cuda12.0)
```

### Source checkout used

This runbook was validated with `nccl-tests` tag `v2.17.2`:
```bash
git fetch --tags
git checkout v2.17.2
```

### Upgrading later (optional)

If you later upgrade the NVIDIA driver to a version compatible with CUDA 12.8/12.9 (e.g., 550+), you can unpin NCCL and move to newer NCCL/CTK:
```bash
sudo apt-mark unhold libnccl2 libnccl-dev
sudo apt-get install -y libnccl2 libnccl-dev
# Then rebuild with CUDA_HOME=/usr/local/cuda-12.8 (or newer) consistently.
```


