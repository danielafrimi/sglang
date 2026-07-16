import torch
import triton.language as tl

from sglang.jit_kernel.fp8_triton_dtype import (
    cuda_capability_uses_fp8_e4b15,
    fp8_dtype_to_triton,
)


def test_cuda_capability_uses_fp8_e4b15_before_sm89():
    assert cuda_capability_uses_fp8_e4b15((8, 0))
    assert cuda_capability_uses_fp8_e4b15((8, 6))
    assert not cuda_capability_uses_fp8_e4b15((8, 9))
    assert not cuda_capability_uses_fp8_e4b15((9, 0))
    assert not cuda_capability_uses_fp8_e4b15((10, 0))


def test_fp8_dtype_to_triton_uses_arch_specific_e4m3_name():
    assert (
        fp8_dtype_to_triton(torch.float8_e4m3fn, cuda_capability=(8, 0))
        == tl.float8e4b15
    )
    assert (
        fp8_dtype_to_triton(torch.float8_e4m3fn, cuda_capability=(8, 9))
        == tl.float8e4nv
    )
    assert (
        fp8_dtype_to_triton(torch.float8_e5m2, cuda_capability=(8, 0))
        == tl.float8e5
    )


def test_fp8_dtype_to_triton_maps_e4m3fnuz():
    assert fp8_dtype_to_triton(torch.float8_e4m3fnuz) == tl.float8e4b8
