from __future__ import annotations

import struct
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from zarr.codecs.blosc import BloscCodec
from zarr.registry import register_codec

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

try:
    from nvidia import nvcomp
except ImportError:  # pragma: no cover
    nvcomp = None

_BLOSC_MAX_OVERHEAD = 16
_BLOSC_FLAG_DOSHUFFLE = 0x01
_BLOSC_FLAG_MEMCPYED = 0x02
_BLOSC_FLAG_DOBITSHUFFLE = 0x04
_BLOSC_FLAG_DONT_SPLIT = 0x10
_BLOSC_COMPFORMAT_ZSTD = 4
_BLOSC_MIN_BUFFERSIZE = 128
_BLOSC_MAX_SPLITS = 16


def _read_i32(source: "cp.ndarray", offset: int) -> int:
    raw = cp.asnumpy(source[offset : offset + 4]).tobytes()
    return int(struct.unpack("<i", raw)[0])


class NvcompBloscCodec(BloscCodec):
    """GPU Blosc decoder (zstd + {bitshuffle, noshuffle}) using nvCOMP."""

    @cached_property
    def _zstd_codec(self) -> nvcomp.Codec:
        assert cp is not None
        assert nvcomp is not None
        device = cp.cuda.Device()
        stream = cp.cuda.get_current_stream()
        return nvcomp.Codec(
            algorithm="Zstd",
            bitstream_kind=nvcomp.BitstreamKind.RAW,
            device_id=device.id,
            cuda_stream=stream.ptr,
        )

    @staticmethod
    def _ensure_supported_dtype(chunk_spec: ArraySpec) -> None:
        dtype = np.dtype(chunk_spec.dtype.to_native_dtype())
        if dtype != np.dtype("float32"):
            raise ValueError(
                "NvcompBloscCodec only supports float32 for GPU decode. "
                f"Got dtype={dtype!s}."
            )

    @staticmethod
    def _bitunshuffle(src: "cp.ndarray", *, typesize: int, n_elements: int) -> "cp.ndarray":
        bits = cp.unpackbits(src, bitorder="little")
        matrix = bits.reshape((typesize * 8, n_elements))
        out_bits = matrix.T.reshape((-1,))
        return cp.packbits(out_bits, bitorder="little")

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        source = chunk_bytes.as_array_like()
        if cp is None or not isinstance(source, cp.ndarray):
            return await super()._decode_single(chunk_bytes, chunk_spec)
        if nvcomp is None:
            raise RuntimeError(
                "NvcompBloscCodec requires `nvidia-nvcomp-cu12` to decode Blosc zstd chunks on GPU."
            )

        if source.size < _BLOSC_MAX_OVERHEAD:
            raise ValueError(
                f"Invalid Blosc payload: expected at least {_BLOSC_MAX_OVERHEAD} bytes, got {source.size}."
            )

        header = cp.asnumpy(source[:_BLOSC_MAX_OVERHEAD]).tobytes()
        _, _, flags, typesize, nbytes, blocksize, cbytes = struct.unpack("<BBBBIII", header)
        if source.size < cbytes:
            raise ValueError(
                f"Invalid Blosc payload: cbytes={cbytes} larger than available bytes={source.size}."
            )
        source = source[:cbytes]

        is_memcpyed = (flags & _BLOSC_FLAG_MEMCPYED) != 0
        do_shuffle = (flags & _BLOSC_FLAG_DOSHUFFLE) != 0
        do_bitshuffle = (flags & _BLOSC_FLAG_DOBITSHUFFLE) != 0
        dont_split = (flags & _BLOSC_FLAG_DONT_SPLIT) != 0
        compformat = (flags & 0xE0) >> 5

        self._ensure_supported_dtype(chunk_spec)

        if do_shuffle:
            raise ValueError("NvcompBloscCodec does not support byte-shuffle Blosc chunks.")
        if compformat != _BLOSC_COMPFORMAT_ZSTD:
            raise ValueError(
                "NvcompBloscCodec only supports Blosc chunks with cname='zstd'. "
                f"Got compformat={compformat}."
            )

        if is_memcpyed:
            if cbytes < _BLOSC_MAX_OVERHEAD + nbytes:
                raise ValueError(
                    "Invalid memcpyed Blosc payload: missing raw bytes after header."
                )
            raw = source[_BLOSC_MAX_OVERHEAD : _BLOSC_MAX_OVERHEAD + nbytes]
            return chunk_spec.prototype.buffer.from_array_like(raw.copy())

        if blocksize == 0:
            raise ValueError("Invalid Blosc payload: blocksize must be > 0.")
        if nbytes % typesize != 0:
            raise ValueError(
                f"Invalid Blosc payload: nbytes={nbytes} is not divisible by typesize={typesize}."
            )

        leftover = nbytes % blocksize
        nblocks = nbytes // blocksize + (1 if leftover else 0)

        bstarts_offset = _BLOSC_MAX_OVERHEAD
        bstarts_size = nblocks * 4
        bstarts_end = bstarts_offset + bstarts_size
        if cbytes < bstarts_end:
            raise ValueError("Invalid Blosc payload: missing block-start table.")

        bstarts = np.frombuffer(
            cp.asnumpy(source[bstarts_offset:bstarts_end]).tobytes(),
            dtype="<i4",
            count=nblocks,
        )

        output = cp.empty((nbytes,), dtype=cp.uint8)

        for bi in range(nblocks):
            bsize = leftover if (bi == nblocks - 1 and leftover > 0) else blocksize
            leftover_block = bi == nblocks - 1 and leftover > 0

            if (
                (not dont_split)
                and (typesize <= _BLOSC_MAX_SPLITS)
                and ((blocksize // typesize) >= _BLOSC_MIN_BUFFERSIZE)
                and (not leftover_block)
            ):
                nsplits = typesize
            else:
                nsplits = 1

            if bsize % nsplits != 0:
                raise ValueError(
                    f"Invalid Blosc payload: blocksize {bsize} not divisible by nsplits {nsplits}."
                )
            neblock = bsize // nsplits

            block_tmp = cp.empty((bsize,), dtype=cp.uint8)
            decode_inputs: list[cp.ndarray] = []
            decode_targets: list[tuple[int, int]] = []

            p = int(bstarts[bi])
            for si in range(nsplits):
                if p < 0 or (p + 4) > cbytes:
                    raise ValueError("Invalid Blosc payload: split header outside payload.")
                csize = _read_i32(source, p)
                p += 4
                if csize < 0 or (p + csize) > cbytes:
                    raise ValueError("Invalid Blosc payload: split data outside payload.")
                split = source[p : p + csize]
                p += csize

                start = si * neblock
                end = start + neblock
                if csize == neblock:
                    block_tmp[start:end] = split
                else:
                    decode_inputs.append(split)
                    decode_targets.append((start, end))

            if decode_inputs:
                decoded = self._zstd_codec.decode(nvcomp.as_arrays(decode_inputs))
                cp.cuda.get_current_stream().synchronize()
                for dec, (start, end) in zip(decoded, decode_targets, strict=True):
                    block_tmp[start:end] = cp.asarray(dec, dtype=cp.uint8)

            if do_bitshuffle:
                if bsize % typesize != 0:
                    raise ValueError(
                        f"Invalid bitshuffle payload: block bytes {bsize} not divisible by typesize {typesize}."
                    )
                n_elements = bsize // typesize
                if n_elements % 8 != 0:
                    raise ValueError(
                        "NvcompBloscCodec only supports bitshuffle blocks with element count divisible by 8."
                    )
                block_out = self._bitunshuffle(
                    block_tmp,
                    typesize=typesize,
                    n_elements=n_elements,
                )
            else:
                block_out = block_tmp

            block_start = bi * blocksize
            output[block_start : block_start + bsize] = block_out

        return chunk_spec.prototype.buffer.from_array_like(output)


register_codec("blosc", NvcompBloscCodec, qualname="zarr.codecs.gpu.NvcompBloscCodec")
