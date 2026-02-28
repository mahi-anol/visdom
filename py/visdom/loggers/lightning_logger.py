"""
PyTorch Lightning callback for gradient norm logging to Visdom.

Usage:
    callback = LightningGradientNormLogger(vis, log_every=10)
    trainer  = pl.Trainer(max_epochs=5, callbacks=[callback])
    trainer.fit(model)
"""

import warnings
import torch
import torch.nn as nn
import pytorch_lightning as pl

from visdom.loggers.base import compute_layer_grad_norm
from visdom.loggers.lightning_profiler import LightningHookProfiler, LightningOpProfiler


class LightningGradientNormLogger(pl.Callback):

    def __init__(
        self,
        vis,
        norm_type=2.0,
        log_every=1,
        env="main",
        win_total="grad_norm_total",
        win_per_layer="grad_norm_per_layer",
        log_per_layer=True,
        hook_profiler=None,
        op_profiler=None,
    ):
        super().__init__()
        self.vis           = vis            # visdom connection used to send plots
        self.norm_type     = norm_type      # which L-norm to use (2.0 = euclidean)
        self.log_every     = log_every      # only send to visdom every N steps
        self.env           = env            # visdom environment name
        self.win_total     = win_total      # visdom window name for the total norm line plot
        self.win_per_layer = win_per_layer  # visdom window name for the per-layer bar chart
        self.log_per_layer = log_per_layer  # whether to also show the per-layer bar chart
        self.hook_profiler = hook_profiler  # optional LightningHookProfiler, or None
        self.op_profiler   = op_profiler    # optional LightningOpProfiler, or None

        self._hooks       = []   # list of hook handles so we can remove them later
        self._layer_norms = {}   # maps layer name -> gradient norm, filled by hooks

    def on_train_start(self, trainer, pl_module):
        if self._hooks:
            warnings.warn("Hooks already attached. Detach first.")
            return
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                # register_hook fires automatically during loss.backward()
                self._hooks.append(param.register_hook(self._make_hook(name)))
        if self.op_profiler is not None:
            self.op_profiler.start()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        step = trainer.global_step  # lightning tracks the global step for us
        if step % self.log_every == 0:
            self._flush(step)
        if self.op_profiler is not None:
            self.op_profiler.step()  # advance the torch.profiler schedule

    def on_train_end(self, trainer, pl_module):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._layer_norms.clear()
        if self.op_profiler is not None:
            self.op_profiler.stop()

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

    def _flush(self, step):
        if not self._layer_norms:
            return

        # combine individual layer norms into one total norm
        total = (
            sum(v ** self.norm_type for v in self._layer_norms.values())
            ** (1.0 / self.norm_type)
        )

        self.vis.line(
            Y=[total],
            X=[step],
            win=self.win_total,
            env=self.env,
            update="append",
            opts={"title": "Total Gradient Norm", "xlabel": "Global Step",
                  "ylabel": f"L{int(self.norm_type)} Norm"},
        )

        if self.log_per_layer:
            names = list(self._layer_norms.keys())  # layer names for the bar chart labels
            self.vis.bar(
                X=[self._layer_norms[n] for n in names],
                win=self.win_per_layer,
                env=self.env,
                opts={"title": f"Per-Layer Norms (step {step})",
                      "rownames": names,
                      "ylabel": f"L{int(self.norm_type)} Norm"},
            )

        self._layer_norms.clear()  # reset so next step starts fresh