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
from zarr.codecs.blosc import BloscCodec
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


def _read_i32(source: "cp.ndarray", offset: int) -> int:
    raw = cp.asnumpy(source[offset : offset + 4]).tobytes()
    return int(struct.unpack("<i", raw)[0])


@dataclass(frozen=True)
class _BloscBlockPlan:
    block_tmp: "cp.ndarray"
    output: "cp.ndarray"
    output_start: int
    bsize: int
    do_bitshuffle: bool
    typesize: int


@dataclass(frozen=True)
class _BloscSplitTarget:
    block_plan: _BloscBlockPlan
    start: int
    stop: int


@dataclass(frozen=True)
class _BloscChunkPlan:
    index: int
    output: "cp.ndarray"
    chunk_spec: ArraySpec
    block_plans: tuple[_BloscBlockPlan, ...]


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

    @staticmethod
    def _bitunshuffle(src: "cp.ndarray", *, typesize: int, n_elements: int) -> "cp.ndarray":
        bits = cp.unpackbits(src, bitorder="little")
        matrix = bits.reshape((typesize * 8, n_elements))
        out_bits = matrix.T.reshape((-1,))
        return cp.packbits(out_bits, bitorder="little")

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

    @staticmethod
    def _finalize_block(block_plan: _BloscBlockPlan) -> "cp.ndarray":
        if not block_plan.do_bitshuffle:
            return block_plan.block_tmp
        if block_plan.bsize % block_plan.typesize != 0:
            raise ValueError(
                "Invalid bitshuffle payload: "
                f"block bytes {block_plan.bsize} not divisible by typesize {block_plan.typesize}."
            )
        n_elements = block_plan.bsize // block_plan.typesize
        if n_elements % 8 != 0:
            raise ValueError(
                "NvcompBloscCodec only supports bitshuffle blocks with element count divisible by 8."
            )
        return NvcompBloscCodec._bitunshuffle(
            block_plan.block_tmp,
            typesize=block_plan.typesize,
            n_elements=n_elements,
        )

    def _build_chunk_plan(
        self,
        *,
        index: int,
        source: "cp.ndarray",
        chunk_spec: ArraySpec,
        decode_inputs: list["cp.ndarray"],
        decode_targets: list[_BloscSplitTarget],
    ) -> _BloscChunkPlan | Buffer:
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
            cp.asnumpy(source[bstarts_offset:bstarts_end]).tobytes(),
            dtype="<i4",
            count=nblocks,
        )

        output = cp.empty((nbytes,), dtype=cp.uint8)
        block_plans: list[_BloscBlockPlan] = []

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
            block_tmp = cp.empty((bsize,), dtype=cp.uint8)
            block_plan = _BloscBlockPlan(
                block_tmp=block_tmp,
                output=output,
                output_start=block_index * blocksize,
                bsize=bsize,
                do_bitshuffle=do_bitshuffle,
                typesize=typesize,
            )
            block_plans.append(block_plan)

            p = int(bstarts[block_index])
            for split_index in range(nsplits):
                if p < 0 or (p + 4) > cbytes:
                    raise ValueError("Invalid Blosc payload: split header outside payload.")
                csize = _read_i32(source, p)
                p += 4
                if csize < 0 or (p + csize) > cbytes:
                    raise ValueError("Invalid Blosc payload: split data outside payload.")
                split = source[p : p + csize]
                p += csize

                start = split_index * neblock
                stop = start + neblock
                if csize == neblock:
                    block_tmp[start:stop] = split
                else:
                    decode_inputs.append(split)
                    decode_targets.append(
                        _BloscSplitTarget(block_plan=block_plan, start=start, stop=stop)
                    )

        return _BloscChunkPlan(
            index=index,
            output=output,
            chunk_spec=chunk_spec,
            block_plans=tuple(block_plans),
        )

    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        batch = list(chunks_and_specs)
        results: list[Buffer | None] = [None] * len(batch)
        decode_inputs: list[cp.ndarray] = []
        decode_targets: list[_BloscSplitTarget] = []
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
                decode_targets=decode_targets,
            )
            if isinstance(plan_or_buffer, _BloscChunkPlan):
                gpu_chunk_plans.append(plan_or_buffer)
            else:
                results[index] = plan_or_buffer

        if decode_inputs:
            decoded_splits = await self._run_nvcomp_zstd_batch(decode_inputs, operation="decode")
            for decoded, decode_target in zip(decoded_splits, decode_targets, strict=True):
                decode_target.block_plan.block_tmp[decode_target.start : decode_target.stop] = decoded

        for chunk_plan in gpu_chunk_plans:
            for block_plan in chunk_plan.block_plans:
                block_out = self._finalize_block(block_plan)
                block_plan.output[
                    block_plan.output_start : block_plan.output_start + block_plan.bsize
                ] = block_out
            results[chunk_plan.index] = chunk_plan.chunk_spec.prototype.buffer.from_array_like(
                chunk_plan.output
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
