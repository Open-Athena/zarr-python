# Using GPUs with Zarr

Zarr can use GPUs to accelerate your workload by running `zarr.Config.enable_gpu`.

!!! note
    GPU codec acceleration is currently available for the Blosc codec when all of
    the following hold:

    - `cname="zstd"`
    - `shuffle` is `bitshuffle` or `noshuffle`
    - array dtype is `float32`

## Reading data into device memory

[`zarr.config`][] configures Zarr to use GPU memory for the data
buffers used internally by Zarr via `enable_gpu()`.

```python
import zarr
import cupy as cp
zarr.config.enable_gpu()
store = zarr.storage.MemoryStore()
z = zarr.create_array(
    store=store, shape=(100, 100), chunks=(10, 10), dtype="float32",
)
type(z[:10, :10])
# cupy.ndarray
```

Note that the output type is a `cupy.ndarray` rather than a NumPy array.
