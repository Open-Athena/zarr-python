from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import numcodecs
import numpy as np
from numcodecs.zstd import Zstd as NumcodecsZstd
from packaging.version import Version

from zarr.abc.codec import BytesBytesCodec
from zarr.codecs.blosc import BloscCname, BloscCodec, BloscShuffle, CName, Shuffle
from zarr.codecs.zstd import parse_checksum, parse_zstd_level
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON, concurrent_map, parse_named_configuration
from zarr.core.config import config
from zarr.registry import register_codec

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

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
_DEFAULT_BLOSC_BITUNSHUFFLE_MAX_BYTES = 100 * 1024 * 1024


def _parse_positive_int(value: int, *, name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int. Got {type(value)} instead.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0. Got {value}.")
    return value


@dataclass(frozen=True)
class _BloscSegmentPlan:
    dest_offset: int
    length: int
    raw_source: "cp.ndarray | None" = None
    decode_index: int | None = None


@dataclass(frozen=True)
class _BloscBitshufflePlan:
    typesize: int
    blocksize: int
    full_block_bytes: int
    tail_bytes: int


@dataclass(frozen=True)
class _BloscChunkPlan:
    index: int
    scratch: "cp.ndarray"
    chunk_spec: ArraySpec
    segments: tuple[_BloscSegmentPlan, ...]
    bitshuffle: _BloscBitshufflePlan | None


class _NvcompZstdMixin:
    def _require_nvcomp(self, *, codec_name: str, operation: Literal["encode", "decode"]) -> None:
        if nvcomp is None:
            raise RuntimeError(
                f"{codec_name} requires `nvidia-nvcomp-cu12` to {operation} GPU-backed zstd chunks."
            )

    @cached_property
    def _nvcomp_zstd_codec(self) -> nvcomp.Codec:
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
    def _coerce_nvcomp_output(array: Any) -> "cp.ndarray":
        out = cp.asarray(array)
        if out.dtype != np.dtype("B"):
            out = out.view(np.dtype("B"))
        return out

    async def _run_nvcomp_zstd_batch(
        self,
        arrays: Iterable["cp.ndarray"],
        *,
        operation: Literal["decode", "encode"],
    ) -> list["cp.ndarray"]:
        array_list = list(arrays)
        if not array_list:
            return []

        outputs = getattr(self._nvcomp_zstd_codec, operation)(nvcomp.as_arrays(array_list))
        event = cp.cuda.Event()
        event.record()
        await asyncio.to_thread(event.synchronize)
        return [self._coerce_nvcomp_output(output) for output in outputs]


@dataclass(frozen=True)
class NvcompZstdCodec(_NvcompZstdMixin, BytesBytesCodec):
    """GPU-backed zstd codec using nvCOMP for batched encode/decode."""

    is_fixed_size = True

    level: int = 0
    checksum: bool = False

    def __init__(self, *, level: int = 0, checksum: bool = False) -> None:
        _numcodecs_version = Version(numcodecs.__version__)
        if _numcodecs_version < Version("0.13.0"):
            raise RuntimeError(
                "numcodecs version >= 0.13.0 is required to use the zstd codec. "
                f"Version {_numcodecs_version} is currently installed."
            )

        object.__setattr__(self, "level", parse_zstd_level(level))
        object.__setattr__(self, "checksum", parse_checksum(checksum))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "zstd")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "zstd", "configuration": {"level": self.level, "checksum": self.checksum}}

    @cached_property
    def _cpu_zstd_codec(self) -> NumcodecsZstd:
        return NumcodecsZstd.from_config({"level": self.level, "checksum": self.checksum})

    async def _decode_single_cpu(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return await asyncio.to_thread(
            as_numpy_array_wrapper,
            self._cpu_zstd_codec.decode,
            chunk_bytes,
            chunk_spec.prototype,
        )

    async def _encode_single_cpu(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return await asyncio.to_thread(
            as_numpy_array_wrapper,
            self._cpu_zstd_codec.encode,
            chunk_bytes,
            chunk_spec.prototype,
        )

    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        batch = list(chunks_and_specs)
        results: list[Buffer | None] = [None] * len(batch)
        gpu_entries: list[tuple[int, cp.ndarray, ArraySpec]] = []
        cpu_entries: list[tuple[int, Buffer, ArraySpec]] = []

        for index, (chunk_bytes, chunk_spec) in enumerate(batch):
            if chunk_bytes is None:
                continue
            source = chunk_bytes.as_array_like()
            if cp is not None and isinstance(source, cp.ndarray):
                gpu_entries.append((index, source, chunk_spec))
            else:
                cpu_entries.append((index, chunk_bytes, chunk_spec))

        if gpu_entries:
            self._require_nvcomp(codec_name=type(self).__name__, operation="decode")
            outputs = await self._run_nvcomp_zstd_batch(
                (source for _, source, _ in gpu_entries), operation="decode"
            )
            for (index, _, chunk_spec), output in zip(gpu_entries, outputs, strict=True):
                results[index] = chunk_spec.prototype.buffer.from_array_like(output)

        if cpu_entries:
            cpu_outputs = await concurrent_map(
                [(chunk_bytes, chunk_spec) for _, chunk_bytes, chunk_spec in cpu_entries],
                self._decode_single_cpu,
                config.get("async.concurrency"),
            )
            for (index, _, _), output in zip(cpu_entries, cpu_outputs, strict=True):
                results[index] = output

        return results

    async def encode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        batch = list(chunks_and_specs)
        results: list[Buffer | None] = [None] * len(batch)
        gpu_entries: list[tuple[int, cp.ndarray, ArraySpec]] = []
        cpu_entries: list[tuple[int, Buffer, ArraySpec]] = []

        for index, (chunk_bytes, chunk_spec) in enumerate(batch):
            if chunk_bytes is None:
                continue
            source = chunk_bytes.as_array_like()
            if cp is not None and isinstance(source, cp.ndarray):
                gpu_entries.append((index, source, chunk_spec))
            else:
                cpu_entries.append((index, chunk_bytes, chunk_spec))

        if gpu_entries:
            self._require_nvcomp(codec_name=type(self).__name__, operation="encode")
            outputs = await self._run_nvcomp_zstd_batch(
                (source for _, source, _ in gpu_entries), operation="encode"
            )
            for (index, _, chunk_spec), output in zip(gpu_entries, outputs, strict=True):
                results[index] = chunk_spec.prototype.buffer.from_array_like(output)

        if cpu_entries:
            cpu_outputs = await concurrent_map(
                [(chunk_bytes, chunk_spec) for _, chunk_bytes, chunk_spec in cpu_entries],
                self._encode_single_cpu,
                config.get("async.concurrency"),
            )
            for (index, _, _), output in zip(cpu_entries, cpu_outputs, strict=True):
                results[index] = output

        return results

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


class NvcompBloscCodec(_NvcompZstdMixin, BloscCodec):
    """GPU Blosc decoder (zstd + {bitshuffle, noshuffle}) using nvCOMP."""

    _default_bitshuffle_max_bytes = _DEFAULT_BLOSC_BITUNSHUFFLE_MAX_BYTES
    bitshuffle_max_bytes: int

    def __init__(
        self,
        *,
        typesize: int | None = None,
        cname: BloscCname | CName = BloscCname.zstd,
        clevel: int = 5,
        shuffle: BloscShuffle | Shuffle | None = None,
        blocksize: int = 0,
        bitshuffle_max_bytes: int | None = None,
    ) -> None:
        super().__init__(
            typesize=typesize,
            cname=cname,
            clevel=clevel,
            shuffle=shuffle,
            blocksize=blocksize,
        )
        if bitshuffle_max_bytes is None:
            bitshuffle_max_bytes = config.get(
                "gpu.blosc_bitshuffle_max_bytes",
                type(self)._default_bitshuffle_max_bytes,
            )
        object.__setattr__(
            self,
            "bitshuffle_max_bytes",
            _parse_positive_int(bitshuffle_max_bytes, name="bitshuffle_max_bytes"),
        )

    @staticmethod
    def _bitunshuffle(src: "cp.ndarray", *, typesize: int, n_elements: int) -> "cp.ndarray":
        bits = cp.unpackbits(src, bitorder="little")
        matrix = bits.reshape((typesize * 8, n_elements))
        out_bits = matrix.T.reshape((-1,))
        return cp.packbits(out_bits, bitorder="little")

    @staticmethod
    def _bitunshuffle_blocks(
        src: "cp.ndarray",
        *,
        typesize: int,
        blocksize: int,
        nblocks: int,
    ) -> "cp.ndarray":
        bits = cp.unpackbits(src, bitorder="little")
        matrix = bits.reshape((nblocks, typesize * 8, blocksize // typesize))
        out_bits = matrix.transpose((0, 2, 1)).reshape((-1,))
        return cp.packbits(out_bits, bitorder="little")

    @cached_property
    def _scatter_segments_kernel(self) -> cp.RawKernel:
        return cp.RawKernel(
            r"""
            extern "C" __global__
            void scatter_segments(
                const unsigned long long* src_ptrs,
                const long long* dst_offsets,
                const long long* lengths,
                unsigned char* out
            ) {
                const long long seg = static_cast<long long>(blockIdx.x);
                const unsigned char* src =
                    reinterpret_cast<const unsigned char*>(src_ptrs[seg]);
                unsigned char* dst = out + dst_offsets[seg];
                const long long len = lengths[seg];
                for (long long i = threadIdx.x; i < len; i += blockDim.x) {
                    dst[i] = src[i];
                }
            }
            """,
            "scatter_segments",
        )

    @staticmethod
    def _split_count(*, typesize: int, blocksize: int, dont_split: bool, leftover_block: bool) -> int:
        if (
            (not dont_split)
            and (typesize <= _BLOSC_MAX_SPLITS)
            and ((blocksize // typesize) >= _BLOSC_MIN_BUFFERSIZE)
            and (not leftover_block)
        ):
            return typesize
        return 1

    def _scatter_segments(
        self,
        chunk_plan: _BloscChunkPlan,
        decoded_splits: list["cp.ndarray"],
    ) -> None:
        if not chunk_plan.segments:
            return

        src_ptrs = np.empty((len(chunk_plan.segments),), dtype=np.uintp)
        dst_offsets = np.empty((len(chunk_plan.segments),), dtype=np.int64)
        lengths = np.empty((len(chunk_plan.segments),), dtype=np.int64)

        for idx, segment in enumerate(chunk_plan.segments):
            if segment.raw_source is not None:
                source = segment.raw_source
            else:
                assert segment.decode_index is not None
                source = decoded_splits[segment.decode_index]
            if source.size != segment.length:
                raise ValueError(
                    "Invalid Blosc payload: split decode size mismatch. "
                    f"Expected {segment.length} bytes, got {source.size}."
                )
            src_ptrs[idx] = source.data.ptr
            dst_offsets[idx] = segment.dest_offset
            lengths[idx] = segment.length

        threads = 256
        self._scatter_segments_kernel(
            (len(chunk_plan.segments),),
            (threads,),
            (
                cp.asarray(src_ptrs),
                cp.asarray(dst_offsets),
                cp.asarray(lengths),
                chunk_plan.scratch,
            ),
        )

    def _apply_bitshuffle(self, chunk_plan: _BloscChunkPlan) -> None:
        bitshuffle = chunk_plan.bitshuffle
        if bitshuffle is None:
            return

        full_block_bytes = bitshuffle.full_block_bytes
        if full_block_bytes > 0:
            window_blocks = max(1, self.bitshuffle_max_bytes // bitshuffle.blocksize)
            window_bytes = window_blocks * bitshuffle.blocksize
            for start in range(0, full_block_bytes, window_bytes):
                stop = min(start + window_bytes, full_block_bytes)
                nblocks = (stop - start) // bitshuffle.blocksize
                chunk_plan.scratch[start:stop] = self._bitunshuffle_blocks(
                    chunk_plan.scratch[start:stop],
                    typesize=bitshuffle.typesize,
                    blocksize=bitshuffle.blocksize,
                    nblocks=nblocks,
                )

        if bitshuffle.tail_bytes == 0:
            return

        tail_start = full_block_bytes
        tail_stop = tail_start + bitshuffle.tail_bytes
        if bitshuffle.tail_bytes % bitshuffle.typesize != 0:
            raise ValueError(
                "Invalid bitshuffle payload: "
                f"block bytes {bitshuffle.tail_bytes} not divisible by typesize {bitshuffle.typesize}."
            )
        n_elements = bitshuffle.tail_bytes // bitshuffle.typesize
        if n_elements % 8 != 0:
            raise ValueError(
                "NvcompBloscCodec only supports bitshuffle blocks with element count divisible by 8."
            )
        chunk_plan.scratch[tail_start:tail_stop] = self._bitunshuffle(
            chunk_plan.scratch[tail_start:tail_stop],
            typesize=bitshuffle.typesize,
            n_elements=n_elements,
        )

    def _build_chunk_plan(
        self,
        *,
        index: int,
        source: "cp.ndarray",
        chunk_spec: ArraySpec,
        decode_inputs: list["cp.ndarray"],
    ) -> _BloscChunkPlan | Buffer:
        if source.size < _BLOSC_MAX_OVERHEAD:
            raise ValueError(
                f"Invalid Blosc payload: expected at least {_BLOSC_MAX_OVERHEAD} bytes, got {source.size}."
            )

        host_source = memoryview(cp.asnumpy(source))
        _, _, flags, typesize, nbytes, blocksize, cbytes = struct.unpack_from(
            "<BBBBIII", host_source, 0
        )
        if source.size < cbytes:
            raise ValueError(
                f"Invalid Blosc payload: cbytes={cbytes} larger than available bytes={source.size}."
            )
        host_source = host_source[:cbytes]
        source = source[:cbytes]

        is_memcpyed = (flags & _BLOSC_FLAG_MEMCPYED) != 0
        do_shuffle = (flags & _BLOSC_FLAG_DOSHUFFLE) != 0
        do_bitshuffle = (flags & _BLOSC_FLAG_DOBITSHUFFLE) != 0
        dont_split = (flags & _BLOSC_FLAG_DONT_SPLIT) != 0
        compformat = (flags & 0xE0) >> 5

        if do_shuffle:
            raise ValueError("NvcompBloscCodec does not support byte-shuffle Blosc chunks.")
        if compformat != _BLOSC_COMPFORMAT_ZSTD:
            raise ValueError(
                "NvcompBloscCodec only supports Blosc chunks with cname='zstd'. "
                f"Got compformat={compformat}."
            )

        if is_memcpyed:
            if cbytes < _BLOSC_MAX_OVERHEAD + nbytes:
                raise ValueError("Invalid memcpyed Blosc payload: missing raw bytes after header.")
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
            host_source[bstarts_offset:bstarts_end],
            dtype="<i4",
            count=nblocks,
        )

        scratch = cp.empty((nbytes,), dtype=cp.uint8)
        segments: list[_BloscSegmentPlan] = []

        for block_index in range(nblocks):
            bsize = leftover if (block_index == nblocks - 1 and leftover > 0) else blocksize
            leftover_block = block_index == nblocks - 1 and leftover > 0
            nsplits = self._split_count(
                typesize=typesize,
                blocksize=bsize,
                dont_split=dont_split,
                leftover_block=leftover_block,
            )
            if bsize % nsplits != 0:
                raise ValueError(
                    f"Invalid Blosc payload: blocksize {bsize} not divisible by nsplits {nsplits}."
                )

            neblock = bsize // nsplits
            p = int(bstarts[block_index])
            block_start = block_index * blocksize
            for split_index in range(nsplits):
                if p < 0 or (p + 4) > cbytes:
                    raise ValueError("Invalid Blosc payload: split header outside payload.")
                csize = int(struct.unpack_from("<i", host_source, p)[0])
                p += 4
                if csize < 0 or (p + csize) > cbytes:
                    raise ValueError("Invalid Blosc payload: split data outside payload.")
                split = source[p : p + csize]
                p += csize

                dest_offset = block_start + split_index * neblock
                if csize == neblock:
                    segments.append(
                        _BloscSegmentPlan(
                            dest_offset=dest_offset,
                            length=neblock,
                            raw_source=split,
                        )
                    )
                else:
                    decode_index = len(decode_inputs)
                    decode_inputs.append(split)
                    segments.append(
                        _BloscSegmentPlan(
                            dest_offset=dest_offset,
                            length=neblock,
                            decode_index=decode_index,
                        )
                    )

        bitshuffle = None
        if do_bitshuffle:
            bitshuffle = _BloscBitshufflePlan(
                typesize=typesize,
                blocksize=blocksize,
                full_block_bytes=(nblocks - (1 if leftover else 0)) * blocksize,
                tail_bytes=leftover,
            )

        return _BloscChunkPlan(
            index=index,
            scratch=scratch,
            chunk_spec=chunk_spec,
            segments=tuple(segments),
            bitshuffle=bitshuffle,
        )

    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        batch = list(chunks_and_specs)
        results: list[Buffer | None] = [None] * len(batch)
        decode_inputs: list[cp.ndarray] = []
        gpu_chunk_plans: list[_BloscChunkPlan] = []
        cpu_entries: list[tuple[int, Buffer, ArraySpec]] = []

        for index, (chunk_bytes, chunk_spec) in enumerate(batch):
            if chunk_bytes is None:
                continue

            source = chunk_bytes.as_array_like()
            if cp is None or not isinstance(source, cp.ndarray):
                cpu_entries.append((index, chunk_bytes, chunk_spec))
                continue

            self._require_nvcomp(codec_name=type(self).__name__, operation="decode")
            plan_or_buffer = self._build_chunk_plan(
                index=index,
                source=source,
                chunk_spec=chunk_spec,
                decode_inputs=decode_inputs,
            )
            if isinstance(plan_or_buffer, _BloscChunkPlan):
                gpu_chunk_plans.append(plan_or_buffer)
            else:
                results[index] = plan_or_buffer

        decoded_splits: list[cp.ndarray] = []
        if decode_inputs:
            decoded_splits = await self._run_nvcomp_zstd_batch(decode_inputs, operation="decode")

        for chunk_plan in gpu_chunk_plans:
            self._scatter_segments(chunk_plan, decoded_splits)
            self._apply_bitshuffle(chunk_plan)
            results[chunk_plan.index] = chunk_plan.chunk_spec.prototype.buffer.from_array_like(
                chunk_plan.scratch
            )

        if cpu_entries:
            cpu_outputs = await concurrent_map(
                [(chunk_bytes, chunk_spec) for _, chunk_bytes, chunk_spec in cpu_entries],
                super()._decode_single,
                config.get("async.concurrency"),
            )
            for (index, _, _), output in zip(cpu_entries, cpu_outputs, strict=True):
                results[index] = output

        return results


register_codec("zstd", NvcompZstdCodec, qualname="zarr.codecs.gpu.NvcompZstdCodec")
register_codec("blosc", NvcompBloscCodec, qualname="zarr.codecs.gpu.NvcompBloscCodec")
