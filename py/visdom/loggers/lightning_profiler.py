"""
Profiling helpers for PyTorch Lightning training loops.

LightningHookProfiler - measures time spent inside each backward hook
LightningOpProfiler   - wraps torch.profiler, started/stopped via trainer callbacks
"""

import time
import torch


class LightningHookProfiler:
    """Records how long each grad-norm computation takes inside a backward hook."""

    def __init__(self):
        self.overhead_ms = []   # list of timings (in ms) for each hook call
        self._t0 = None         # timestamp saved at the start of a measurement

    def record_start(self):
        self._t0 = time.perf_counter()  # save the current time before norm is computed

    def record_end(self):
        if self._t0 is not None:
            elapsed = (time.perf_counter() - self._t0) * 1000  # convert seconds to ms
            self.overhead_ms.append(elapsed)
            self._t0 = None

    def reset(self):
        self.overhead_ms.clear()
        self._t0 = None

    def summary(self):
        if not self.overhead_ms:
            return {}
        data = self.overhead_ms
        return {
            "count":    len(data),                  # total number of hook calls recorded
            "mean_ms":  sum(data) / len(data),       # average time per hook call
            "p95_ms":    sorted(data)[max(0, int(len(data) * 0.95) - 1)],
            "max_ms":   max(data),                   # slowest single hook call
            "total_ms": sum(data),                   # total time spent in all hooks
        }


class LightningOpProfiler:
    """
    torch.profiler wrapper for Lightning. Unlike TorchOpProfiler, it uses
    start()/stop() instead of a context manager because Lightning manages
    the training lifecycle through callbacks.

        profiler = LightningOpProfiler()
        callback = LightningGradientNormLogger(vis, op_profiler=profiler)
        trainer  = pl.Trainer(callbacks=[callback])
        trainer.fit(model)

        print(profiler.summary())
    """

    def __init__(self, wait=1, warmup=1, active=5, log_dir="./profiler_logs/lightning"):
        self._wait    = wait      # steps to skip before profiling starts
        self._warmup  = warmup    # steps to warm up (data collected but not saved)
        self._active  = active    # steps to actually record
        self._log_dir = log_dir   # where to write the tensorboard trace files
        self._profiler = None     # the actual torch.profiler object, set on start()

    def start(self):
        # always profile CPU; add CUDA only if a GPU is available
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=self._wait, warmup=self._warmup, active=self._active
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self._log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            activities=activities,
        )
        self._profiler.__enter__()

    def step(self):
        # call this once per optimizer step so the profiler can advance its schedule
        if self._profiler is not None:
            self._profiler.step()

    def stop(self):
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)

    def summary(self, top_n=15):
        if self._profiler is None:
            return "Profiler has not been started yet."
        return self._profiler.key_averages().table(
            sort_by="cpu_time_total", row_limit=top_n
        )

    def export_chrome_trace(self, path):
        if self._profiler is None:
            raise RuntimeError("Run the profiler first before exporting.")
        self._profiler.export_chrome_trace(path)