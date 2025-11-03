# NVLink Runtime Failure Simulation Guide

## Overview

This modification adds runtime NVLink failure simulation functionality to NCCL AllReduce tests, designed to test NCCL library's handling capabilities when NVLink connections are suddenly interrupted.

## Features

### 1. Runtime Failure Injection
- Dynamically disable GPU-to-GPU P2P connections at specified test iterations
- Simulate real-world NVLink hardware failure scenarios
- Support precise failure timing control

### 2. Detailed Monitoring and Logging
- Real-time P2P connection status monitoring
- NCCL error handling behavior tracking
- Custom kernel failure detection
- Performance impact analysis

### 3. Multiple Test Scenarios
- Failure response testing for different deviceImpl implementations
- Early/mid/late failure injection
- Performance comparison before and after failures

## Environment Variable Configuration

### `NCCL_NVLINK_FAILURE_ITERATION`
Controls at which iteration to inject the failure:
```bash
export NCCL_NVLINK_FAILURE_ITERATION=10  # Inject failure at iteration 10
export NCCL_NVLINK_FAILURE_ITERATION=-1  # Disable failure simulation (default)
```

### Recommended NCCL Debug Settings
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_P2P_DISABLE=0  # Ensure P2P is initially enabled
```

## Usage

### 1. Basic Usage
```bash
# Build the tests
make -j$(nproc)

# Inject NVLink failure at iteration 5
export NCCL_NVLINK_FAILURE_ITERATION=5
./build/all_reduce_perf -b 32M -e 32M -f 2 -g 4 -n 20 -d 0
```

### 2. Using Test Script
```bash
# Run complete failure simulation test suite
./test_nvlink_failure.sh
```

### 3. Manual Testing of Different Scenarios
```bash
# Test NCCL built-in implementation failure handling
export NCCL_NVLINK_FAILURE_ITERATION=8
./build/all_reduce_perf -b 64M -e 64M -g 8 -n 15 -d 0

# Test custom LSA kernel failure response
export NCCL_NVLINK_FAILURE_ITERATION=5
./build/all_reduce_perf -b 32M -e 32M -g 4 -n 10 -d 1

# Test multimem kernel failure handling
export NCCL_NVLINK_FAILURE_ITERATION=3
./build/all_reduce_perf -b 128M -e 128M -g 4 -n 12 -d 3
```

## Log Analysis

### Key Log Identifiers

#### `[NVLink Failure Simulation]`
Main logs from the failure simulation system:
```
[NVLink Failure Simulation] Will inject failure at iteration 10
[NVLink Failure Simulation] Injecting failure at iteration 10/20
[NVLink Failure Simulation] NVLink failure injected successfully
```

#### `[P2P Status Check]`
P2P connection status monitoring:
```
[P2P Status Check] Checking P2P connectivity between GPUs...
[P2P Status] GPU 0 -> GPU 1: ENABLED
[P2P Status] GPU 0 -> GPU 2: DISABLED
[P2P Status Check] Active connections: 8/12 (66.7%)
```

#### `[NVLink Failure Monitor]`
Runtime failure monitoring:
```
[NVLink Failure Monitor] Running AllReduce with deviceImpl=0 after failure injection
[NVLink Failure Monitor] NCCL AllReduce succeeded despite NVLink failure - NCCL handled gracefully
[NVLink Failure Monitor] Attempting LSA kernel after P2P failure - may fail
```

### Expected Behavior

#### deviceImpl=0 (NCCL Built-in)
- **Normal case**: NCCL should detect P2P failure and automatically switch to backup communication paths
- **Success indicator**: See "NCCL handled gracefully" message
- **Performance impact**: Bandwidth may drop significantly, but test should complete

#### deviceImpl=1-2 (LSA Kernels)
- **Possible outcome**: Kernels may fail as they directly depend on P2P access
- **Failure indicators**: CUDA errors or kernel timeouts
- **Expected behavior**: Should see clear error messages

#### deviceImpl=3-4 (Multimem Kernels)
- **Hardware dependency**: Requires Hopper+ architecture support
- **Failure handling**: May be more robust than LSA kernels, but still may fail

## Recommended Test Scenarios

### 1. Failure Timing Tests
```bash
# Early failure (immediately after warmup)
export NCCL_NVLINK_FAILURE_ITERATION=1

# Mid-test failure (halfway through test)
export NCCL_NVLINK_FAILURE_ITERATION=10

# Late failure (near end of test)
export NCCL_NVLINK_FAILURE_ITERATION=18
```

### 2. Message Size Impact
```bash
# Small messages (may be more latency dependent)
./build/all_reduce_perf -b 1K -e 1M -f 2

# Large messages (more bandwidth dependent)
./build/all_reduce_perf -b 32M -e 512M -f 2
```

### 3. GPU Count Scaling
```bash
# 2 GPUs (simple P2P)
./build/all_reduce_perf -g 2

# 4 GPUs (complex topology)
./build/all_reduce_perf -g 4

# 8 GPUs (multi-tier topology)
./build/all_reduce_perf -g 8
```

### 4. Emulation of failure cases for different message sizes
```bash
export NCCL_NVLINK_FAILURE_MIN_BYTES=$((128*1024*1024)) && ./build/all_reduce_perf -b 8M -e 512M -f 2 -g 2
```

## Troubleshooting

### Common Issues

#### 1. P2P Disable Failure
```
[NVLink Failure] Warning: Failed to disable P2P access: peer access was not enabled
```
**Solution**: This is normal - some GPU pairs may not have P2P enabled initially

#### 2. Custom Kernels Unavailable
```
Test failure common.cu:XXX
```
**Solution**: Ensure NCCL version >= 2.28.0 and GPU supports required features

#### 3. No Failure Injection Observed
**Check items**:
- Confirm `NCCL_NVLINK_FAILURE_ITERATION` is set
- Failure iteration number is less than total iterations
- System actually has multiple GPUs

### Debugging Tips

1. **Enable verbose logging**:
   ```bash
   export NCCL_DEBUG=TRACE
   export CUDA_LAUNCH_BLOCKING=1
   ```

2. **Check GPU topology**:
   ```bash
   nvidia-smi topo -m
   ```

3. **Monitor GPU usage**:
   ```bash
   nvidia-smi -l 1
   ```

## Experimental Value

This failure simulation functionality helps with:

1. **NCCL Robustness Verification**: Test NCCL's recovery capabilities during hardware failures
2. **Performance Benchmarking**: Quantify the impact of NVLink failures on performance
3. **Failure Detection Latency**: Measure NCCL's time to detect and respond to failures
4. **Backup Path Efficiency**: Evaluate performance of backup communication paths like PCIe/InfiniBand
5. **Application Adaptability**: Test upper-layer application tolerance to communication failures

## Future Extensions

Future enhancements could include:
- Partial failure simulation (disable only certain connections)
- Failure recovery testing (re-enable P2P)
- Network failure simulation (InfiniBand interruption)
- Automatic failure detection and reporting
- Performance regression analysis

make clean && NCCL_HOME=/home/zx/nccl/build make -j$(nproc)

LD_LIBRARY_PATH=/home/zx/nccl/build/lib:$LD_LIBRARY_PATH ./build/all_reduce_perf -b 32M -e 512M -f 2 -g 2

NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL LD_LIBRARY_PATH=/home/zx/nccl/build/lib:$LD_LIBRARY_PATH ./build/all_reduce_perf -b 32M -e 512M -f 2 -g 2 2>&1 | tee nccl_debug.log
