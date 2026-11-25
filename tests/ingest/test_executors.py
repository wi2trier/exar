import asyncio
import os
import subprocess
import tempfile
import time
from concurrent.futures import CancelledError
from pathlib import Path

import pytest

from exar.ingest import ProcessPool, WorkerCrashed
from exar.ingest.parsers import _run


def test_failures_take_down_only_their_worker(tmp_path: Path) -> None:
    with ProcessPool(max_workers=1, timeout=1) as pool:
        with pytest.raises(ValueError):
            pool.submit(int, "x").result()

        with pytest.raises(WorkerCrashed):
            pool.submit(os._exit, 3).result()

        start = time.monotonic()

        with pytest.raises(TimeoutError):
            pool.submit(
                subprocess.run,
                ["sh", "-c", f"echo $$ > {tmp_path}/pid && exec sleep 30"],
            ).result()

        # The process the task started is killed with its worker.
        time.sleep(0.5)

        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "pid").read_text()), 0)

        running = pool.submit(time.sleep, 30)
        time.sleep(0.5)

        assert running.cancel()

        with pytest.raises(CancelledError):
            running.result()

        assert pool.submit(pow, 2, 8).result() == 256
        assert time.monotonic() - start < 10


def test_converters_die_with_their_worker(tmp_path: Path) -> None:
    script = f"echo $$ > {tmp_path}/pid && exec sleep 30"

    with ProcessPool(max_workers=1, timeout=1) as pool, pytest.raises(TimeoutError):
        pool.submit(_run, ["sh", "-c", script], timeout=300).result()

    time.sleep(0.5)

    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "pid").read_text()), 0)


def test_worker_temporary_files_go_with_the_worker() -> None:
    script = 'touch "$TMPDIR/partial" && exec sleep 30'

    with ProcessPool(max_workers=1, timeout=1) as pool:
        tmpdir = Path(pool.submit(tempfile.gettempdir).result())

        assert tmpdir.is_dir()

        with pytest.raises(TimeoutError):
            pool.submit(subprocess.run, ["sh", "-c", script]).result()

        assert not tmpdir.exists()


def test_max_tasks_recycles_workers() -> None:
    with ProcessPool(1) as warm, ProcessPool(1, max_tasks=1) as isolated:
        assert len({warm.submit(os.getpid).result() for _ in range(2)}) == 1
        assert len({isolated.submit(os.getpid).result() for _ in range(2)}) == 2


async def test_stdlib_compatibility() -> None:
    with ProcessPool(max_workers=2) as pool:
        loop = asyncio.get_running_loop()

        assert await loop.run_in_executor(pool, pow, 2, 4) == 16
        assert list(pool.map(pow, [2, 3], [2, 2])) == [4, 9]
