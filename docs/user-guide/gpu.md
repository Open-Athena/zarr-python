# Using GPUs with Zarr

Zarr can use GPUs to accelerate your workload by enabling GPU buffers and codecs
with `zarr.config.enable_gpu()`.

!!! note
    With `enable_gpu()`:

    - array reads return `cupy.ndarray` values
    - the default Zarr v3 `zstd` codec is replaced with a GPU-backed nvCOMP
      implementation
    - Blosc chunks can be decoded on the GPU when `cname="zstd"` and `shuffle`
      is `bitshuffle` or `noshuffle`

    Blosc GPU acceleration currently applies to decode only.

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
