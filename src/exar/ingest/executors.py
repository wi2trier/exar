"""An executor that runs crash-prone native code, such as parsers and media decoders, in worker processes.

`ProcessPool` runs picklable callables in worker processes, so a native crash or a hang takes down a worker, never the caller.
Unlike the pool of the standard library, a crashing worker fails only its own task and is replaced,
a task running longer than the timeout of the pool is killed with its worker,
and cancelling a running task kills its worker too.
Workers stay warm with their loaded models, or are recycled after `max_tasks` tasks,
where `max_tasks=1` runs every task in a fresh process, so one malformed input cannot corrupt the next.
Each worker gets a temporary directory of its own as `TMPDIR`, removed once the worker is gone,
so what a killed task and its converters leave behind never outlives them.
The workers isolate crashes and hangs, not exploits: they run with the privileges of the caller,
which unpickles their replies, so input that takes over a worker takes over the caller too.
It is a `concurrent.futures.Executor`, so it plugs into `loop.run_in_executor` and the executor parameters of cbrkit.
"""

import os
import pickle
import shutil
import signal
import sys
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import get_context
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess
from queue import Empty, SimpleQueue
from typing import Any

if sys.platform == "win32":
    from multiprocessing.connection import PipeConnection as Connection
else:
    from multiprocessing.connection import Connection

__all__ = [
    "WORKER_TMP_PREFIX",
    "ProcessPool",
    "WorkerCrashed",
    "limit_threads",
    "limit_worker",
]

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
# The platform default, forkserver on Linux, which imports the caller's main module
# once instead of in every worker, and never forks the threaded caller itself.
_CONTEXT = get_context()

WORKER_TMP_PREFIX = "exar-worker-"
"""What starts the name of the temporary directory of a worker, which a crash of the caller leaves behind.

The name is this prefix, the pid of the caller that created it, a dash, and a random suffix,
so a sweep can tell the directories of live callers from those of dead ones.
"""

_READY = b"ready"
"""What a worker sends once its initializer ran, before it takes its first task."""

type _Task = tuple[Future[Any], Callable[[], Any]]
# The value of a task, or the exception it raised.
type _Outcome = tuple[Any, BaseException | None]


class WorkerCrashed(Exception):
    """A worker process exited without returning the result of its task."""


def limit_threads(threads: int) -> None:
    """Caps the native thread pools of this process.

    Native libraries such as torch, BLAS, and tokenizers use every core by default,
    so several processes would oversubscribe them.
    The environment variables reach the libraries loaded afterwards, and an imported torch is capped directly.
    """
    os.environ.update(dict.fromkeys(_THREAD_VARIABLES, str(threads)))

    if (torch := sys.modules.get("torch")) is not None:
        torch.set_num_threads(threads)


def limit_worker(threads: int, max_image_pixels: int | None = None) -> None:
    """Caps the native thread pools and the image size of this process, an initializer of `ProcessPool`.

    Args:
        threads: Threads of each native thread pool, see `limit_threads`.
        max_image_pixels: Pixels of an image Pillow decodes, such as one embedded in an Office document,
            the decompression bomb limit of Pillow if None.
    """
    limit_threads(threads)

    if max_image_pixels is not None:
        import PIL.Image

        PIL.Image.MAX_IMAGE_PIXELS = max_image_pixels


def _resolve(future: Future[Any], outcome: _Outcome) -> None:
    """Sets the outcome of a future unless it was cancelled meanwhile."""
    value, error = outcome

    with suppress(InvalidStateError):
        if error is None:
            future.set_result(value)
        else:
            future.set_exception(error)


def _serve(
    connection: Connection, initializer: Callable[[], object] | None, tmpdir: str
) -> None:
    """Runs the tasks the pool sends until it closes the connection, the main loop of a worker."""
    # A process group of its own, which the pool kills with the worker.
    os.setpgid(0, 0)
    # For the processes it starts, and for this one, whose default may be cached already.
    os.environ["TMPDIR"] = tmpdir
    tempfile.tempdir = tmpdir

    if initializer is not None:
        initializer()

    connection.send_bytes(_READY)

    while True:
        try:
            data = connection.recv_bytes()
        except EOFError:
            return

        try:
            outcome: _Outcome = (pickle.loads(data)(), None)
        except Exception as error:  # noqa: BLE001
            outcome = (None, error)

        try:
            connection.send(outcome)
        except Exception as error:  # noqa: BLE001
            # The result or exception of the task did not pickle.
            connection.send((None, error))


@dataclass(slots=True)
class _Worker:
    process: BaseProcess
    connection: Connection
    tmpdir: str
    tasks: int = 0
    _reaped: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def signal(self) -> None:
        """Kills the worker with the processes it started, such as pandoc, which share its process group.

        Only before the worker is reaped, since its id names no group of it afterwards,
        and may name an unrelated one once the system reuses it.
        """
        with self._lock:
            if self._reaped or (pid := self.process.pid) is None:
                return

            # macOS refuses with EPERM once only the exited worker is left.
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)

            self.process.kill()

    def kill(self) -> None:
        """Kills the worker and its group, then reaps it and removes its temporary directory."""
        self.signal()

        with self._lock:
            self.process.join()
            self._reaped = True

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def stop(self) -> None:
        """Lets an idle worker exit by closing its connection, killing it if it does not.

        The exit is awaited on its sentinel, which leaves it unreaped, so a group it leaves behind is still killed.
        """
        self.connection.close()
        _ = wait([self.process.sentinel], 1)
        self.kill()

    def run(self, task: _Task, timeout: float | None) -> _Outcome:
        """Runs a task, killing the worker on a timeout, a crash, or the cancellation of its future."""
        future, call = task
        self.tasks += 1
        future.add_done_callback(lambda done: done.cancelled() and self.signal())

        try:
            self.connection.send(call)

            if not self.connection.poll(timeout):
                self.kill()

                return None, TimeoutError(f"The task ran longer than {timeout}s")

            return self.connection.recv()
        except EOFError, OSError:
            self.kill()

            return None, WorkerCrashed(
                f"The worker exited with code {self.process.exitcode}"
            )
        except Exception as error:  # noqa: BLE001
            # The call or its result did not pickle, which leaves the worker intact.
            return None, error


@dataclass(slots=True, eq=False)
class ProcessPool(Executor):
    """A pool of worker processes that survives crashes and hangs of its tasks.

    A normal exception of a task is raised as is, a crash as `WorkerCrashed`, and a timeout as `TimeoutError`.
    A crashed, timed out, or cancelled task takes its worker down, which the next task replaces.
    Futures stay pending while their task runs, so that `cancel` also stops a running task by killing its worker.
    The callable must be importable by its qualified name, its arguments and result picklable.
    Workers are daemonic, so they never outlive the caller, but cannot start processes of `multiprocessing` themselves.

    Args:
        max_workers: Processes that run at once.
        timeout: Seconds a task may run before its worker is killed, unlimited if None.
        max_tasks: Tasks a worker runs before it is replaced, unlimited if None, 1 for a fresh process per task.
        initializer: Runs in every worker before its first task, such as `limit_worker`.

    Examples:
        >>> with ProcessPool(1) as pool:
        ...     pool.submit(pow, 2, 10).result()
        1024
    """

    max_workers: int
    timeout: float | None = None
    max_tasks: int | None = None
    initializer: Callable[[], object] | None = None
    _queue: SimpleQueue[_Task | None] = field(
        default_factory=SimpleQueue, init=False, repr=False
    )
    _threads: list[threading.Thread] = field(
        default_factory=list, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _closed: bool = field(default=False, init=False, repr=False)

    def submit[**P, R](
        self, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[R]:
        """Schedules a call on the next free worker."""
        future: Future[R] = Future()

        with self._lock:
            if self._closed:
                raise RuntimeError("cannot schedule new futures after shutdown")

            self._queue.put((future, partial(fn, *args, **kwargs)))

            if len(self._threads) < self.max_workers:
                thread = threading.Thread(target=self._supervise, daemon=True)
                thread.start()
                self._threads.append(thread)

        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Stops accepting tasks and stops the workers once the queued tasks are done.

        Args:
            wait: Whether to block until the queued tasks are done and the workers stopped.
            cancel_futures: Whether to cancel the queued tasks that did not start.
        """
        with self._lock:
            self._closed = True

            while cancel_futures:
                try:
                    task = self._queue.get_nowait()
                except Empty:
                    break

                if task is not None:
                    task[0].cancel()

            for _ in self._threads:
                self._queue.put(None)

        if wait:
            for thread in self._threads:
                thread.join()

    def _start(self) -> _Worker:
        """A new worker whose initializer ran, within the timeout of a task.

        Raises:
            TimeoutError: If the initializer outlives the timeout, after the worker is killed.
            WorkerCrashed: If the worker exits before it is ready.
        """
        tmpdir = tempfile.mkdtemp(prefix=f"{WORKER_TMP_PREFIX}{os.getpid()}-")
        connection, child = _CONTEXT.Pipe()
        process = _CONTEXT.Process(
            target=_serve, args=(child, self.initializer, tmpdir), daemon=True
        )

        try:
            process.start()
        except BaseException:
            shutil.rmtree(tmpdir, ignore_errors=True)

            raise

        child.close()
        worker = _Worker(process, connection, tmpdir)

        try:
            ready = connection.poll(self.timeout)

            if ready:
                _ = connection.recv_bytes()

        # Its connection closed, mostly by an initializer that raised.
        except EOFError, OSError:
            worker.kill()

            raise WorkerCrashed(
                f"The worker exited with code {process.exitcode} before it was ready"
            ) from None

        except BaseException:
            worker.kill()

            raise

        if not ready:
            worker.kill()

            raise TimeoutError(f"The worker did not start within {self.timeout}s")

        return worker

    def _supervise(self) -> None:
        """Feeds the queued tasks to one worker, replacing it when it went down or is used up."""
        worker: _Worker | None = None

        try:
            while (task := self._queue.get()) is not None:
                future = task[0]

                if future.cancelled():
                    continue

                if worker is None:
                    try:
                        worker = self._start()
                    except Exception as error:  # noqa: BLE001
                        _resolve(future, (None, error))

                        continue

                _resolve(future, worker.run(task, self.timeout))

                if (
                    worker.process.exitcode is not None
                    or future.cancelled()
                    or worker.tasks == self.max_tasks
                ):
                    worker.stop()
                    worker = None
        finally:
            if worker is not None:
                worker.stop()
