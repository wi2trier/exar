import asyncio
import os
import stat
from pathlib import Path

import pytest

from exar.helpers import (
    atomic_write,
    bounded_as_completed,
    bounded_gather,
    shield_to_completion,
)


def test_atomic_write_keeps_permissions_whatever_the_umask(tmp_path: Path) -> None:
    path = tmp_path / "shared.txt"
    path.write_text("old")
    path.chmod(0o666)
    umask = os.umask(0o077)

    try:
        atomic_write(path, "new")
    finally:
        os.umask(umask)

    assert stat.S_IMODE(path.stat().st_mode) == 0o666


def test_shield_to_completion_finishes_then_reraises_cancel() -> None:
    done: list[bool] = []

    async def work() -> None:
        await asyncio.sleep(0.05)
        done.append(True)

    async def run() -> None:
        task = asyncio.ensure_future(shield_to_completion(work()))
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    assert done == [True]


def test_shield_to_completion_propagates_work_error() -> None:
    async def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        asyncio.run(shield_to_completion(boom()))


def test_bounded_as_completed_stays_within_limit() -> None:
    in_flight: list[int] = []
    peak: list[int] = [0]

    async def double(item: int) -> int:
        in_flight.append(item)
        peak[0] = max(peak[0], len(in_flight))
        await asyncio.sleep(0.01)
        in_flight.remove(item)

        return 2 * item

    async def run() -> list[int]:
        async with bounded_as_completed(range(10), double, limit=3) as results:
            return [x async for x in results]

    assert sorted(asyncio.run(run())) == [2 * i for i in range(10)]
    assert peak[0] <= 3


def test_bounded_as_completed_unwinds_in_flight_on_cancel() -> None:
    started: list[int] = []
    unwound: list[int] = []

    async def block(item: int) -> int:
        started.append(item)

        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            unwound.append(item)
            raise

        return item

    async def consume() -> None:
        async with bounded_as_completed(range(5), block, limit=2) as results:
            async for _ in results:
                pass

    async def run() -> None:
        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.02)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    # Only the two calls that got a slot ran, and each unwound before the generator returned.
    assert started == unwound == [0, 1]


def test_bounded_as_completed_unwinds_when_the_block_raises() -> None:
    unwound: list[int] = []

    async def block(item: int) -> int:
        if item == 0:
            return item

        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            unwound.append(item)
            raise

        return item

    async def run() -> None:
        async with bounded_as_completed(range(3), block, limit=3) as results:
            async for _ in results:
                raise ValueError("stop")

    with pytest.raises(ValueError, match="stop"):
        asyncio.run(run())

    # Unwound before the error left the block, not later by the garbage collector.
    assert sorted(unwound) == [1, 2]


def test_bounded_helpers_refuse_a_limit_below_one() -> None:
    async def double(item: int) -> int:
        return 2 * item

    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(bounded_gather([1], double, limit=0))


def test_shield_to_completion_logs_an_error_after_cancel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fail_later() -> None:
        await asyncio.sleep(0.05)
        raise ValueError("lost")

    async def run() -> None:
        task = asyncio.ensure_future(shield_to_completion(fail_later()))
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError) as raised:
            await task

        assert isinstance(raised.value.__context__, ValueError)

    asyncio.run(run())

    assert "lost" in caplog.text
