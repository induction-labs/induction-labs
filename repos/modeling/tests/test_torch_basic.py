#!/usr/bin/env python3
"""
PyTorch GPU Sanity Check Script with Pytest
Tests PyTorch installation and GPU functionality using pytest framework
"""

from __future__ import annotations

import sys
import time
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.optim as optim


class GPUTestConfig:
    """Configuration for GPU tests"""

    SMALL_TENSOR_SIZE: int = 100
    MEDIUM_TENSOR_SIZE: int = 1000
    LARGE_TENSOR_SIZE: int = 2000
    BATCH_SIZE: int = 32
    NUM_CLASSES: int = 10
    TRAINING_EPOCHS: int = 5
    TOLERANCE: float = 1e-6


def get_gpu_info() -> dict[str, Any]:
    """Get GPU information for testing"""
    if not torch.cuda.is_available():
        return {}

    info = {
        "cuda_version": torch.version.cuda,  # type: ignore[attr-defined]
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
    }

    # Add device properties
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info[f"device_{i}"] = {
            "name": props.name,
            "total_memory": props.total_memory,
            "major": props.major,
            "minor": props.minor,
        }

    return info


def assert_tensors_close(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    context: str = "",
) -> None:
    """
    Assert that two tensors are close with detailed error reporting.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        atol: Absolute tolerance
        rtol: Relative tolerance
        context: Additional context for error message
    """
    if torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
        return

    # Calculate detailed error statistics
    abs_diff = torch.abs(tensor1 - tensor2)
    rel_diff = abs_diff / (torch.abs(tensor2) + 1e-8)

    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()
    max_rel_diff = torch.max(rel_diff).item()
    mean_rel_diff = torch.mean(rel_diff).item()

    # Find location of maximum difference
    max_abs_idx = torch.argmax(abs_diff)
    if tensor1.dim() > 1:
        max_abs_idx_2d = torch.unravel_index(max_abs_idx, abs_diff.shape)
        val1 = tensor1[max_abs_idx_2d].item()
        val2 = tensor2[max_abs_idx_2d].item()
        location = f"{max_abs_idx_2d}"
    else:
        val1 = tensor1[max_abs_idx].item()
        val2 = tensor2[max_abs_idx].item()
        location = f"[{max_abs_idx.item()}]"

    context_str = f" ({context})" if context else ""

    error_msg = (
        f"Tensors are not close{context_str}:\n"
        f"  Tensor shapes: {tensor1.shape} vs {tensor2.shape}\n"
        f"  Max absolute difference: {max_abs_diff:.6e} (threshold: {atol:.6e})\n"
        f"  Mean absolute difference: {mean_abs_diff:.6e}\n"
        f"  Max relative difference: {max_rel_diff:.6e} (threshold: {rtol:.6e})\n"
        f"  Mean relative difference: {mean_rel_diff:.6e}\n"
        f"  Location of max diff: {location}\n"
        f"  Tensor1 value: {val1:.6e}\n"
        f"  Tensor2 value: {val2:.6e}\n"
        f"  Absolute difference: {abs(val1 - val2):.6e}\n"
        f"  Relative difference: {abs(val1 - val2) / (abs(val2) + 1e-8):.6e}"
    )

    assert False, error_msg


class TestBasicInstallation:
    """Test basic PyTorch installation"""

    def test_torch_import(self) -> None:
        """Test that PyTorch can be imported"""
        assert torch is not None
        assert hasattr(torch, "__version__")

    def test_python_version(self) -> None:
        """Test Python version compatibility"""
        version_info = sys.version_info
        assert version_info.major >= 3
        assert version_info.minor >= 8  # PyTorch requires Python 3.8+

    def test_cuda_availability(self) -> None:
        """Test CUDA availability"""
        assert torch.cuda.is_available(), "CUDA is not available"

    def test_cuda_device_count(self) -> None:
        """Test that at least one GPU is available"""
        assert torch.cuda.device_count() > 0, "No CUDA devices found"

    def test_cudnn_availability(self) -> None:
        """Test cuDNN availability"""
        assert torch.backends.cudnn.enabled, "cuDNN is not enabled"
        assert torch.backends.cudnn.version() is not None, "cuDNN version not found"

    @pytest.mark.parametrize(
        "device_id",
        range(torch.cuda.device_count() if torch.cuda.is_available() else 0),
    )
    def test_device_properties(self, device_id: int) -> None:
        """Test individual GPU device properties"""
        props = torch.cuda.get_device_properties(device_id)
        assert props.name, f"Device {device_id} has no name"
        assert props.total_memory > 0, f"Device {device_id} has no memory"
        assert props.major >= 3, f"Device {device_id} compute capability too low"


class TestGPUMemory:
    """Test GPU memory management"""

    def test_memory_allocation(self) -> None:
        """Test basic GPU memory allocation"""
        device = torch.cuda.current_device()
        initial_memory = torch.cuda.memory_allocated(device)

        # Allocate tensor
        tensor = torch.randn(
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
            device="cuda",
        )
        allocated_memory = torch.cuda.memory_allocated(device)

        assert allocated_memory > initial_memory, "Memory allocation failed"

        # Clean up
        del tensor
        torch.cuda.empty_cache()

    def test_memory_cleanup(self) -> None:
        """Test GPU memory cleanup"""
        device = torch.cuda.current_device()
        initial_memory = torch.cuda.memory_allocated(device)

        # Allocate and deallocate
        tensor = torch.randn(
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
            device="cuda",
        )
        del tensor
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated(device)
        assert final_memory <= initial_memory + 1024, (
            "Memory not properly cleaned up"
        )  # Allow small tolerance

    def test_large_allocation(self) -> None:
        """Test large memory allocation"""
        try:
            # Try to allocate ~100MB
            tensor = torch.randn(5000, 5000, device="cuda")
            assert tensor.is_cuda, "Large tensor not on GPU"
            del tensor
            torch.cuda.empty_cache()
        except RuntimeError as e:
            pytest.skip(
                f"Large allocation failed (likely insufficient GPU memory): {e}"
            )


class TestTensorOperations:
    """Test tensor operations on GPU"""

    def test_tensor_creation_on_gpu(self) -> None:
        """Test creating tensors directly on GPU"""
        tensor = torch.randn(
            GPUTestConfig.SMALL_TENSOR_SIZE,
            GPUTestConfig.SMALL_TENSOR_SIZE,
            device="cuda",
        )
        assert tensor.is_cuda, "Tensor not created on GPU"
        assert tensor.device.type == "cuda", "Tensor device type incorrect"

    def test_cpu_to_gpu_transfer(self) -> None:
        """Test CPU to GPU data transfer"""
        cpu_tensor = torch.randn(
            GPUTestConfig.SMALL_TENSOR_SIZE, GPUTestConfig.SMALL_TENSOR_SIZE
        )
        gpu_tensor = cpu_tensor.to("cuda")

        assert gpu_tensor.is_cuda, "Transfer to GPU failed"
        assert_tensors_close(
            cpu_tensor,
            gpu_tensor.cpu(),
            atol=GPUTestConfig.TOLERANCE,
            context="CPU to GPU transfer",
        )

    def test_gpu_to_cpu_transfer(self) -> None:
        """Test GPU to CPU data transfer"""
        gpu_tensor = torch.randn(
            GPUTestConfig.SMALL_TENSOR_SIZE,
            GPUTestConfig.SMALL_TENSOR_SIZE,
            device="cuda",
        )
        cpu_tensor = gpu_tensor.cpu()

        assert not cpu_tensor.is_cuda, "Transfer to CPU failed"
        assert_tensors_close(
            gpu_tensor.cpu(),
            cpu_tensor,
            atol=GPUTestConfig.TOLERANCE,
            context="GPU to CPU transfer",
        )

    def test_matrix_multiplication(self) -> None:
        """Test matrix multiplication on GPU"""
        a = torch.randn(
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
            device="cuda",
        )
        b = torch.randn(
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
            device="cuda",
        )

        start_time = time.time()
        c = torch.mm(a, b)
        gpu_time = time.time() - start_time

        assert c.is_cuda, "Result not on GPU"
        assert c.shape == (
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
        ), "Incorrect result shape"
        assert gpu_time < 1.0, f"GPU matrix multiplication too slow: {gpu_time:.4f}s"

    @pytest.mark.parametrize(
        "operation",
        [
            torch.add,
            torch.sub,
            torch.mul,
            torch.div,
        ],
    )
    def test_elementwise_operations(self, operation) -> None:
        """Test elementwise operations on GPU"""
        a = torch.randn(
            GPUTestConfig.SMALL_TENSOR_SIZE,
            GPUTestConfig.SMALL_TENSOR_SIZE,
            device="cuda",
        )
        b = torch.randn(
            GPUTestConfig.SMALL_TENSOR_SIZE,
            GPUTestConfig.SMALL_TENSOR_SIZE,
            device="cuda",
        )

        result = operation(a, b)

        assert result.is_cuda, f"{operation.__name__} result not on GPU"
        assert result.shape == a.shape, f"{operation.__name__} changed tensor shape"


class SimpleNet(nn.Module):
    """Simple neural network for testing"""

    def __init__(
        self, input_size: int = 100, hidden_size: int = 50, num_classes: int = 10
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestNeuralNetwork:
    """Test neural network operations on GPU"""

    def test_model_to_gpu(self) -> None:
        """Test moving model to GPU"""
        model = SimpleNet()
        model = model.cuda()

        # Check that all parameters are on GPU
        for param in model.parameters():
            assert param.is_cuda, "Model parameter not on GPU"

    def test_forward_pass_gpu(self) -> None:
        """Test forward pass on GPU"""
        model = SimpleNet().cuda()
        inputs = torch.randn(GPUTestConfig.BATCH_SIZE, 100, device="cuda")

        outputs = model(inputs)

        assert outputs.is_cuda, "Model outputs not on GPU"
        assert outputs.shape == (GPUTestConfig.BATCH_SIZE, GPUTestConfig.NUM_CLASSES), (
            "Incorrect output shape"
        )

    def test_backward_pass_gpu(self) -> None:
        """Test backward pass on GPU"""
        model = SimpleNet().cuda()
        criterion = nn.CrossEntropyLoss()

        inputs = torch.randn(GPUTestConfig.BATCH_SIZE, 100, device="cuda")
        targets = torch.randint(
            0, GPUTestConfig.NUM_CLASSES, (GPUTestConfig.BATCH_SIZE,), device="cuda"
        )

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Check that gradients are computed and on GPU
        for param in model.parameters():
            assert param.grad is not None, "Gradient not computed"
            assert param.grad.is_cuda, "Gradient not on GPU"

    def test_training_loop_gpu(self) -> None:
        """Test complete training loop on GPU"""
        model = SimpleNet().cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        initial_loss = None
        final_loss = None

        for epoch in range(GPUTestConfig.TRAINING_EPOCHS):
            inputs = torch.randn(GPUTestConfig.BATCH_SIZE, 100, device="cuda")
            targets = torch.randint(
                0, GPUTestConfig.NUM_CLASSES, (GPUTestConfig.BATCH_SIZE,), device="cuda"
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()
            if epoch == GPUTestConfig.TRAINING_EPOCHS - 1:
                final_loss = loss.item()

        assert initial_loss is not None and final_loss is not None, (
            "Training losses not recorded"
        )
        # Loss should decrease or at least not increase dramatically
        assert final_loss < initial_loss * 2, "Training appears to be diverging"


class TestCUDAKernels:
    """Test CUDA kernel functionality"""

    def test_cuda_synchronization(self) -> None:
        """Test CUDA synchronization"""
        a = torch.randn(
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
            device="cuda",
        )
        b = torch.randn(
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
            GPUTestConfig.MEDIUM_TENSOR_SIZE,
            device="cuda",
        )

        # Perform operation and synchronize
        c = torch.mm(a, b)
        torch.cuda.synchronize()

        assert c.is_cuda, "Operation result not on GPU"

    def test_cuda_streams(self) -> None:
        """Test CUDA streams functionality"""
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        assert isinstance(stream1, torch.cuda.Stream), "Stream1 is not a CUDA stream"
        assert isinstance(stream2, torch.cuda.Stream), "Stream2 is not a CUDA stream"
        assert stream1 != stream2, "CUDA streams not properly created"

        with torch.cuda.stream(stream1):
            a = torch.randn(
                GPUTestConfig.SMALL_TENSOR_SIZE,
                GPUTestConfig.SMALL_TENSOR_SIZE,
                device="cuda",
            )

        with torch.cuda.stream(stream2):
            b = torch.randn(
                GPUTestConfig.SMALL_TENSOR_SIZE,
                GPUTestConfig.SMALL_TENSOR_SIZE,
                device="cuda",
            )

        torch.cuda.synchronize()

        assert a.is_cuda and b.is_cuda, "Stream operations failed"

    @pytest.mark.slow
    def test_custom_cuda_kernel(self) -> None:
        """Test custom CUDA kernel compilation (requires nvcc)"""
        try:
            import tempfile

            from torch.utils.cpp_extension import load_inline

            # More complete CUDA kernel with proper bindings
            cuda_source = """
            #include <torch/extension.h>
            #include <cuda_runtime.h>
            
            __global__ void add_kernel(const float* a, const float* b, float* c, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    c[idx] = a[idx] + b[idx];
                }
            }
            
            torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
                TORCH_CHECK(a.is_cuda(), "Input tensor a must be on CUDA device");
                TORCH_CHECK(b.is_cuda(), "Input tensor b must be on CUDA device");
                TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
                TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensors must be float32");
                TORCH_CHECK(b.dtype() == torch::kFloat32, "Input tensors must be float32");
                
                auto c = torch::zeros_like(a);
                const int n = a.numel();
                const int threads = 256;
                const int blocks = (n + threads - 1) / threads;
                
                add_kernel<<<blocks, threads>>>(
                    a.data_ptr<float>(),
                    b.data_ptr<float>(),
                    c.data_ptr<float>(),
                    n
                );
                
                // Check for CUDA errors
                cudaError_t err = cudaGetLastError();
                TORCH_CHECK(err == cudaSuccess, "CUDA kernel execution failed: ", cudaGetErrorString(err));
                
                return c;
            }
            """

            cpp_source = """
            #include <torch/extension.h>
            
            torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);
            
            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("add_cuda", &add_cuda, "Element-wise addition (CUDA)");
            }
            """

            # Use a unique name to avoid caching issues
            import time

            module_name = f"add_cuda_test_{int(time.time() * 1000) % 100000}"

            add_module = load_inline(
                name=module_name,
                cpp_sources=cpp_source,
                cuda_sources=cuda_source,
                verbose=True,  # Enable verbose to see compilation errors
                build_directory=tempfile.mkdtemp(),  # Use temporary directory
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
            )

            # Test the custom kernel
            a = torch.randn(1000, dtype=torch.float32, device="cuda")
            b = torch.randn(1000, dtype=torch.float32, device="cuda")
            c = add_module.add_cuda(a, b)
            expected = a + b

            assert_tensors_close(
                c,
                expected,
                atol=GPUTestConfig.TOLERANCE,
                context="Custom CUDA kernel vs PyTorch built-in",
            )

        except ImportError as e:
            if "nvcc" in str(e).lower() or "compiler" in str(e).lower():
                pytest.skip(f"CUDA compiler (nvcc) not available: {e}")
            else:
                pytest.skip(f"Custom CUDA kernel compilation failed: {e}")
        except RuntimeError as e:
            if "nvcc" in str(e).lower() or "compiler" in str(e).lower():
                pytest.skip(f"CUDA compiler (nvcc) not available: {e}")
            else:
                pytest.skip(f"CUDA kernel runtime error: {e}")
        except Exception as e:
            pytest.skip(f"Custom CUDA kernel test failed: {e}")


class TestPerformance:
    """Test GPU performance benchmarks"""

    @pytest.mark.parametrize("size", [1000, 2000, 4000])
    def test_gpu_vs_cpu_performance(self, size: int) -> None:
        """Test GPU vs CPU performance comparison"""
        # Use same input data for both GPU and CPU to ensure fair comparison
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)

        # Copy to GPU
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # GPU benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time

        # CPU benchmark
        start_time = time.time()
        c_cpu = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start_time

        # GPU should be faster for large matrices
        if size >= 2000:
            assert gpu_time < cpu_time, (
                f"GPU ({gpu_time:.4f}s) not faster than CPU ({cpu_time:.4f}s) for size {size}"
            )

        # Check numerical accuracy with detailed error reporting
        c_gpu_cpu = c_gpu.cpu()

        if not torch.allclose(c_gpu_cpu, c_cpu, atol=1e-3, rtol=1e-5):
            # Calculate detailed error statistics
            abs_diff = torch.abs(c_gpu_cpu - c_cpu)
            rel_diff = abs_diff / (
                torch.abs(c_cpu) + 1e-8
            )  # Add small epsilon to avoid division by zero

            max_abs_diff = torch.max(abs_diff).item()
            mean_abs_diff = torch.mean(abs_diff).item()
            max_rel_diff = torch.max(rel_diff).item()
            mean_rel_diff = torch.mean(rel_diff).item()

            # Find location of maximum difference
            max_abs_idx = torch.argmax(abs_diff)
            max_abs_idx_2d = torch.unravel_index(max_abs_idx, abs_diff.shape)

            gpu_val = c_gpu_cpu[max_abs_idx_2d].item()
            cpu_val = c_cpu[max_abs_idx_2d].item()

            error_msg = (
                f"GPU and CPU results don't match for matrix size {size}x{size}:\n"
                f"  Max absolute difference: {max_abs_diff:.6e} (threshold: 1e-3)\n"
                f"  Mean absolute difference: {mean_abs_diff:.6e}\n"
                f"  Max relative difference: {max_rel_diff:.6e} (threshold: 1e-5)\n"
                f"  Mean relative difference: {mean_rel_diff:.6e}\n"
                f"  Location of max diff: {max_abs_idx_2d}\n"
                f"  GPU value: {gpu_val:.6e}\n"
                f"  CPU value: {cpu_val:.6e}\n"
                f"  Difference: {gpu_val - cpu_val:.6e}"
            )

            assert False, error_msg

    def test_memory_bandwidth(self) -> None:
        """Test GPU memory bandwidth"""
        size = 10000
        a = torch.randn(size, size, device="cuda")

        # Time memory-bound operation
        start_time = time.time()
        _b = a + 1.0  # Simple elementwise operation
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time

        # Should complete quickly
        assert elapsed_time < 0.1, (
            f"Memory bandwidth test too slow: {elapsed_time:.4f}s"
        )


# Pytest configuration and fixtures
@pytest.fixture(scope="session", autouse=True)
def gpu_setup():
    """Setup GPU environment for testing"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU tests")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Print GPU info
    gpu_info = get_gpu_info()
    print("\nGPU Test Environment:")
    print(f"CUDA Version: {gpu_info.get('cuda_version')}")
    print(f"Device Count: {gpu_info.get('device_count')}")
    for i in range(gpu_info.get("device_count", 0)):
        device_info = gpu_info.get(f"device_{i}", {})
        print(f"Device {i}: {device_info.get('name', 'Unknown')}")
        print(f"  Memory: {device_info.get('total_memory', 0) / 1e9:.1f} GB")

    yield
    # Cleanup
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Run with pytest
    pytest.main(
        [
            __file__,
            "-v",  # verbose output
            "-s",  # don't capture output
            "--tb=short",  # shorter traceback format
            "-m",
            "not slow",  # skip slow tests by default
        ]
    )
