import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest

import zarr
from zarr.core.sync import (
    SyncError,
    SyncMixin,
    _get_executor,
    _get_lock,
    _get_loop,
    _run_loop_forever,
    _set_gpu_device,
    cleanup_resources,
    loop,
    sync,
)


@pytest.fixture(params=[True, False])
def sync_loop(request: pytest.FixtureRequest) -> asyncio.AbstractEventLoop | None:
    if request.param is True:
        return _get_loop()
    else:
        return None


@pytest.fixture
def clean_state():
    # use this fixture to make sure no existing threads/loops exist in zarr.core.sync
    cleanup_resources()
    yield
    cleanup_resources()


def test_get_loop() -> None:
    # test that calling _get_loop() twice returns the same loop
    loop = _get_loop()
    loop2 = _get_loop()
    assert loop is loop2


def test_get_lock() -> None:
    # test that calling _get_lock() twice returns the same lock
    lock = _get_lock()
    lock2 = _get_lock()
    assert lock is lock2


def test_sync(sync_loop: asyncio.AbstractEventLoop | None) -> None:
    foo = AsyncMock(return_value="foo")
    assert sync(foo(), loop=sync_loop) == "foo"
    foo.assert_awaited_once()


def test_sync_raises(sync_loop: asyncio.AbstractEventLoop | None) -> None:
    foo = AsyncMock(side_effect=ValueError("foo-bar"))
    with pytest.raises(ValueError, match="foo-bar"):
        sync(foo(), loop=sync_loop)
    foo.assert_awaited_once()


def test_sync_timeout() -> None:
    duration = 0.02

    async def foo() -> None:
        await asyncio.sleep(duration)

    with pytest.raises(asyncio.TimeoutError):
        sync(foo(), timeout=duration / 10)


def test_sync_raises_if_no_coroutine(sync_loop: asyncio.AbstractEventLoop | None) -> None:
    def foo() -> str:
        return "foo"

    with pytest.raises(TypeError):
        sync(foo(), loop=sync_loop)  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited")
def test_sync_raises_if_loop_is_closed() -> None:
    loop = _get_loop()

    foo = AsyncMock(return_value="foo")
    with patch.object(loop, "is_closed", return_value=True):
        with pytest.raises(RuntimeError):
            sync(foo(), loop=loop)
    foo.assert_not_awaited()


@pytest.mark.filterwarnings("ignore:Unclosed client session:ResourceWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited")
def test_sync_raises_if_calling_sync_from_within_a_running_loop(
    sync_loop: asyncio.AbstractEventLoop | None,
) -> None:
    def foo() -> str:
        # technically, this should be an async function but doing that
        # yields a warning because it is never awaited by the inner function
        return "foo"

    async def bar() -> str:
        return sync(foo(), loop=sync_loop)  # type: ignore[arg-type]

    with pytest.raises(SyncError):
        sync(bar(), loop=sync_loop)


@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited")
def test_sync_raises_if_loop_is_invalid_type() -> None:
    foo = AsyncMock(return_value="foo")
    with pytest.raises(TypeError):
        sync(foo(), loop=1)  # type: ignore[arg-type]
    foo.assert_not_awaited()


def test_sync_mixin(sync_loop) -> None:
    class AsyncFoo:
        def __init__(self) -> None:
            pass

        async def foo(self) -> str:
            return "foo"

        async def bar(self) -> AsyncGenerator:
            for i in range(10):
                yield i

    class SyncFoo(SyncMixin):
        def __init__(self, async_foo: AsyncFoo) -> None:
            self._async_foo = async_foo

        def foo(self) -> str:
            return self._sync(self._async_foo.foo())

        def bar(self) -> list[int]:
            return self._sync_iter(self._async_foo.bar())

    async_foo = AsyncFoo()
    foo = SyncFoo(async_foo)
    assert foo.foo() == "foo"
    assert foo.bar() == list(range(10))


@pytest.mark.parametrize("workers", [None, 1, 2])
def test_threadpool_executor(clean_state, workers: int | None) -> None:
    with zarr.config.set({"threading.max_workers": workers}):
        _ = zarr.zeros(shape=(1,))  # trigger executor creation
        assert loop != [None]  # confirm loop was created
        if workers is None:
            # confirm no executor was created if no workers were specified
            # (this is the default behavior)
            assert loop[0]._default_executor is None
        else:
            # confirm executor was created and attached to loop as the default executor
            # note: python doesn't have a direct way to get the default executor so we
            # use the private attribute
            assert _get_executor() is loop[0]._default_executor
            assert _get_executor()._max_workers == workers


def test_cleanup_resources_idempotent() -> None:
    _get_executor()  # trigger resource creation (iothread, loop, thread-pool)
    cleanup_resources()
    cleanup_resources()


def test_get_executor_uses_current_gpu_device_for_initializer(
    clean_state, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class FakeLoop:
        def set_default_executor(self, executor: object) -> None:
            captured["default_executor"] = executor

    class FakeExecutor:
        def __init__(
            self,
            *,
            max_workers: int | None,
            thread_name_prefix: str,
            initializer: object,
            initargs: tuple[object, ...],
        ) -> None:
            captured["max_workers"] = max_workers
            captured["thread_name_prefix"] = thread_name_prefix
            captured["initializer"] = initializer
            captured["initargs"] = initargs

        def shutdown(self, wait: bool, cancel_futures: bool) -> None:
            captured["shutdown"] = (wait, cancel_futures)

    monkeypatch.setattr("zarr.core.sync._get_loop", lambda: FakeLoop())
    monkeypatch.setattr("zarr.core.sync.ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr("zarr.core.sync._get_current_gpu_device_id", lambda: 1)

    executor = _get_executor()

    assert captured["default_executor"] is executor
    assert captured["initializer"] is _set_gpu_device
    assert captured["initargs"] == (1,)


def test_get_loop_binds_io_thread_to_current_gpu_device(
    clean_state, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class FakeLoop:
        def call_soon_threadsafe(self, callback: object) -> None:
            captured["callback"] = callback

        def stop(self) -> None:
            captured["stopped"] = True

        def close(self) -> None:
            captured["closed"] = True

    class FakeThread:
        daemon: bool

        def __init__(self, *, target: object, args: tuple[object, ...], name: str) -> None:
            captured["target"] = target
            captured["args"] = args
            captured["name"] = name

        def start(self) -> None:
            captured["started"] = True

        def join(self, timeout: float | None = None) -> None:
            captured["joined"] = timeout

        def is_alive(self) -> bool:
            return False

    fake_loop = FakeLoop()
    monkeypatch.setattr("zarr.core.sync.asyncio.new_event_loop", lambda: fake_loop)
    monkeypatch.setattr("zarr.core.sync.threading.Thread", FakeThread)
    monkeypatch.setattr("zarr.core.sync._get_current_gpu_device_id", lambda: 1)

    observed_loop = _get_loop()

    assert observed_loop is fake_loop
    assert captured["target"] is _run_loop_forever
    assert captured["args"] == (fake_loop, 1)
    assert captured["name"] == "zarr_io"
    assert captured["started"] is True
