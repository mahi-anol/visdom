"""
Hook-based gradient norm logger for plain PyTorch training loops.

Attaches a backward hook to each trainable parameter and sends the
gradient norms to Visdom after every N optimizer steps.

Usage:
    logger = GradientNormLogger(vis, model, log_every=10)
    logger.attach()

    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        logger.step()
        optimizer.step()

    logger.detach()
"""

import warnings
import torch
import torch.nn as nn

from visdom.loggers.base import compute_layer_grad_norm
from visdom.loggers.torch_profiler import HookProfiler, TorchOpProfiler


class GradientNormLogger:

    def __init__(
        self,
        vis,
        model,
        log_every=1,
        norm_type=2.0,
        env="main",
        win_total="grad_norm_total",
        win_per_layer="grad_norm_per_layer",
        log_per_layer=True,
        hook_profiler=None,
        op_profiler=None,
    ):
        self.vis           = vis            # visdom connection used to send plots
        self.model         = model          # the model we are monitoring
        self.log_every     = log_every      # only send to visdom every N steps
        self.norm_type     = norm_type      # which L-norm to use (2.0 = euclidean)
        self.env           = env            # visdom environment name
        self.win_total     = win_total      # visdom window name for the total norm line plot
        self.win_per_layer = win_per_layer  # visdom window name for the per-layer bar chart
        self.log_per_layer = log_per_layer  # whether to also show the per-layer bar chart
        self.hook_profiler = hook_profiler  # optional HookProfiler, or None
        self.op_profiler   = op_profiler    # optional TorchOpProfiler, or None

        self._hooks       = []   # list of hook handles so we can remove them later
        self._step        = 0    # counts how many times step() has been called
        self._layer_norms = {}   # maps layer name -> gradient norm, filled by hooks

    def attach(self):
        if self._hooks:
            warnings.warn("Hooks already attached. Call detach() first.")
            return self
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # register_hook fires automatically during loss.backward()
                self._hooks.append(param.register_hook(self._make_hook(name)))
        return self

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._layer_norms.clear()

    def step(self):
        self._step += 1
        if self.op_profiler is not None:
            self.op_profiler.step()  # advance the torch.profiler schedule
        if self._step % self.log_every == 0:
            self._flush()

    def _make_hook(self, name):
        # returns a closure that captures the layer name
        def hook(grad):
            if self.hook_profiler is not None:
                self.hook_profiler.record_start()
                norm = compute_layer_grad_norm(grad, self.norm_type)
                self.hook_profiler.record_end()
            else:
                norm = compute_layer_grad_norm(grad, self.norm_type)
            self._layer_norms[name] = norm  # store so _flush() can read it
        return hook

    def _flush(self):
        if not self._layer_norms:
            return

        # combine individual layer norms into one total norm
        total = (
            sum(v ** self.norm_type for v in self._layer_norms.values())
            ** (1.0 / self.norm_type)
        )

        self.vis.line(
            Y=[total],
            X=[self._step],
            win=self.win_total,
            env=self.env,
            update="append",
            opts={"title": "Total Gradient Norm", "xlabel": "Step",
                  "ylabel": f"L{int(self.norm_type)} Norm"},
        )

        if self.log_per_layer:
            names = list(self._layer_norms.keys())  # layer names for the bar chart labels
            self.vis.bar(
                X=[self._layer_norms[n] for n in names],
                win=self.win_per_layer,
                env=self.env,
                opts={"title": f"Per-Layer Norms (step {self._step})",
                      "rownames": names,
                      "ylabel": f"L{int(self.norm_type)} Norm"},
            )

        self._layer_norms.clear()  # reset so next step starts fresh

    def __enter__(self):
        return self.attach()

    def __exit__(self, *args):
        self.detach()