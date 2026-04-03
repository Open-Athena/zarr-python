from __future__ import annotations

import warnings

import numpy as np
import pytest

import zarr
from zarr.codecs import NvcompBloscCodec
from zarr.codecs.blosc import BloscCodec
from zarr.errors import ZarrUserWarning
from zarr.testing.utils import gpu_test


def test_nvcomp_blosc_bitshuffle_max_bytes_validation() -> None:
    codec = NvcompBloscCodec(bitshuffle_max_bytes=1024)
    assert codec.bitshuffle_max_bytes == 1024

    with pytest.raises(ValueError):
        NvcompBloscCodec(bitshuffle_max_bytes=0)

    with pytest.raises(TypeError):
        NvcompBloscCodec(bitshuffle_max_bytes="1024")  # type: ignore[arg-type]


@gpu_test
@pytest.mark.parametrize("shuffle", ["bitshuffle", "noshuffle"])
def test_nvcomp_blosc_decode_supported(shuffle: str) -> None:
    import cupy as cp

    src = np.arange(256, dtype=np.float32).reshape(16, 16)
    store = zarr.storage.MemoryStore()
    z = zarr.create_array(
        store=store,
        shape=src.shape,
        chunks=(8, 8),
        dtype=src.dtype,
        compressors=BloscCodec(cname="zstd", shuffle=shuffle),
    )
    z[:, :] = src

    with zarr.config.enable_gpu(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ZarrUserWarning)
        zr = zarr.open_array(store=store, mode="r")
        out = zr[:, :]

    assert isinstance(out, cp.ndarray)
    cp.testing.assert_array_equal(out, cp.asarray(src))


@gpu_test
def test_nvcomp_blosc_decode_raises_on_byte_shuffle() -> None:
    src = np.arange(64, dtype=np.float32).reshape(8, 8)
    store = zarr.storage.MemoryStore()
    z = zarr.create_array(
        store=store,
        shape=src.shape,
        chunks=(8, 8),
        dtype=src.dtype,
        compressors=BloscCodec(cname="zstd", shuffle="shuffle"),
    )
    z[:, :] = src

    with zarr.config.enable_gpu(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ZarrUserWarning)
        zr = zarr.open_array(store=store, mode="r")
        with pytest.raises(ValueError, match="byte-shuffle"):
            _ = zr[:, :]


@gpu_test
def test_nvcomp_blosc_decode_supported_non_float32() -> None:
    import cupy as cp

    src = np.arange(64, dtype=np.float64).reshape(8, 8)
    store = zarr.storage.MemoryStore()
    z = zarr.create_array(
        store=store,
        shape=src.shape,
        chunks=(8, 8),
        dtype=src.dtype,
        compressors=BloscCodec(cname="zstd", shuffle="bitshuffle"),
    )
    z[:, :] = src

    with zarr.config.enable_gpu(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ZarrUserWarning)
        zr = zarr.open_array(store=store, mode="r")
        out = zr[:, :]

    assert isinstance(out, cp.ndarray)
    cp.testing.assert_array_equal(out, cp.asarray(src))


@gpu_test
def test_nvcomp_blosc_decode_raises_on_non_zstd() -> None:
    src = np.arange(64, dtype=np.float32).reshape(8, 8)
    store = zarr.storage.MemoryStore()
    z = zarr.create_array(
        store=store,
        shape=src.shape,
        chunks=(8, 8),
        dtype=src.dtype,
        compressors=BloscCodec(cname="lz4", shuffle="noshuffle"),
    )
    z[:, :] = src

    with zarr.config.enable_gpu(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ZarrUserWarning)
        zr = zarr.open_array(store=store, mode="r")
        with pytest.raises(ValueError, match="cname='zstd'"):
            _ = zr[:, :]


@gpu_test
def test_nvcomp_blosc_decode_batches_multi_block_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cupy as cp

    calls = 0
    original = NvcompBloscCodec._run_nvcomp_zstd_batch

    async def counted(
        self: NvcompBloscCodec,
        arrays: object,
        *,
        operation: str,
    ) -> list[object]:
        nonlocal calls
        calls += 1
        return await original(self, arrays, operation=operation)

    monkeypatch.setattr(NvcompBloscCodec, "_run_nvcomp_zstd_batch", counted)

    src = np.arange(2048, dtype=np.float32)
    store = zarr.storage.MemoryStore()
    z = zarr.create_array(
        store=store,
        shape=src.shape,
        chunks=(2048,),
        dtype=src.dtype,
        compressors=BloscCodec(cname="zstd", shuffle="noshuffle", blocksize=256),
    )
    z[:] = src

    with zarr.config.enable_gpu(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ZarrUserWarning)
        zr = zarr.open_array(store=store, mode="r")
        out = zr[:]

    assert calls == 1
    assert isinstance(out, cp.ndarray)
    cp.testing.assert_array_equal(out, cp.asarray(src))


@gpu_test
def test_nvcomp_blosc_decode_avoids_full_chunk_host_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cupy as cp

    original = cp.asnumpy
    copied_sizes: list[int] = []

    def counted(array: object) -> object:
        size = int(getattr(array, "size", -1))
        copied_sizes.append(size)
        if size > 1024:
            raise AssertionError(f"unexpected large host copy of {size} bytes")
        return original(array)

    monkeypatch.setattr(cp, "asnumpy", counted)

    rng = np.random.default_rng(0)
    src = rng.standard_normal(8192, dtype=np.float32)
    store = zarr.storage.MemoryStore()
    z = zarr.create_array(
        store=store,
        shape=src.shape,
        chunks=(8192,),
        dtype=src.dtype,
        compressors=BloscCodec(cname="zstd", shuffle="noshuffle", blocksize=256),
    )
    z[:] = src

    with zarr.config.enable_gpu(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ZarrUserWarning)
        zr = zarr.open_array(store=store, mode="r")
        out = zr[:]

    assert copied_sizes
    assert isinstance(out, cp.ndarray)
    np.testing.assert_array_equal(original(out), src)


@gpu_test
def test_nvcomp_blosc_decode_batches_multi_chunk_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cupy as cp

    calls = 0
    scatter_calls = 0
    original = NvcompBloscCodec._run_nvcomp_zstd_batch
    original_scatter = NvcompBloscCodec._scatter_segments

    async def counted(
        self: NvcompBloscCodec,
        arrays: object,
        *,
        operation: str,
    ) -> list[object]:
        nonlocal calls
        calls += 1
        return await original(self, arrays, operation=operation)

    def counted_scatter(
        self: NvcompBloscCodec,
        chunk_plan: object,
        decoded_splits: object,
    ) -> None:
        nonlocal scatter_calls
        scatter_calls += 1
        original_scatter(self, chunk_plan, decoded_splits)

    monkeypatch.setattr(NvcompBloscCodec, "_run_nvcomp_zstd_batch", counted)
    monkeypatch.setattr(NvcompBloscCodec, "_scatter_segments", counted_scatter)

    src = np.arange(4096, dtype=np.float32).reshape(64, 64)
    store = zarr.storage.MemoryStore()
    z = zarr.create_array(
        store=store,
        shape=src.shape,
        chunks=(16, 16),
        dtype=src.dtype,
        compressors=BloscCodec(cname="zstd", shuffle="noshuffle", blocksize=256),
    )
    z[:, :] = src

    with zarr.config.enable_gpu(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ZarrUserWarning)
        zr = zarr.open_array(store=store, mode="r")
        out = zr[:, :]

    assert calls == 1
    assert scatter_calls == z.nchunks
    assert isinstance(out, cp.ndarray)
    cp.testing.assert_array_equal(out, cp.asarray(src))


@gpu_test
def test_nvcomp_blosc_bitshuffle_windowing_batches_full_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cupy as cp

    calls = 0
    original = NvcompBloscCodec._bitunshuffle_blocks

    def counted(
        src: object,
        *,
        typesize: int,
        blocksize: int,
        nblocks: int,
    ) -> object:
        nonlocal calls
        calls += 1
        return original(src, typesize=typesize, blocksize=blocksize, nblocks=nblocks)

    monkeypatch.setattr(NvcompBloscCodec, "_bitunshuffle_blocks", staticmethod(counted))

    src = np.arange(1024, dtype=np.float32)
    store = zarr.storage.MemoryStore()
    z = zarr.create_array(
        store=store,
        shape=src.shape,
        chunks=(1024,),
        dtype=src.dtype,
        compressors=BloscCodec(cname="zstd", shuffle="bitshuffle", blocksize=256),
    )
    z[:] = src

    with zarr.config.enable_gpu(), zarr.config.set(
        {"gpu.blosc_bitshuffle_max_bytes": 512}
    ), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ZarrUserWarning)
        zr = zarr.open_array(store=store, mode="r")
        out = zr[:]

    assert calls == src.nbytes // 512
    assert isinstance(out, cp.ndarray)
    cp.testing.assert_array_equal(out, cp.asarray(src))
