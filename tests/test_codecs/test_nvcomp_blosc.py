from __future__ import annotations

import warnings

import numpy as np
import pytest

import zarr
from zarr.codecs import NvcompBloscCodec
from zarr.codecs.blosc import BloscCodec
from zarr.errors import ZarrUserWarning
from zarr.testing.utils import gpu_test


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
def test_nvcomp_blosc_decode_batches_multi_chunk_frames(
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
    assert isinstance(out, cp.ndarray)
    cp.testing.assert_array_equal(out, cp.asarray(src))
