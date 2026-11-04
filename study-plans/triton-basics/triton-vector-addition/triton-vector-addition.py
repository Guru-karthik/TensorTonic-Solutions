import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    # 1. Identify which program this is
    pid = tl.program_id(axis=0)
    
    # 2. Calculate the array indices (offsets) this program is responsible for
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 3. Create a mask to protect against out-of-bounds access on the last block
    mask = offsets < n
    
    # 4. Calculate the memory addresses by adding offsets to the base pointers
    x_ptrs = x_ptr + offsets
    y_ptrs = y_ptr + offsets
    out_ptrs = out_ptr + offsets
    
    # 5. Load the data from memory into registers (using the mask)
    x_vals = tl.load(x_ptrs, mask=mask)
    y_vals = tl.load(y_ptrs, mask=mask)
    
    # 6. Perform the element-wise addition
    out_vals = x_vals + y_vals
    
    # 7. Write the results back to memory (using the mask)
    tl.store(out_ptrs, out_vals, mask=mask)


def solve(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch vector_add_kernel on the provided tensors."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)