import sys
import unittest
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from visdom.loggers.lightning_profiler import LightningHookProfiler, LightningOpProfiler

try:
    import pytorch_lightning as pl
    from visdom.loggers.lightning_logger import LightningGradientNormLogger
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False

skip_no_lightning = pytest.mark.skipif(
    not _PL_AVAILABLE, reason="pytorch_lightning not installed"
)


def _vis():
    v = MagicMock()
    v.line = MagicMock(return_value="win")
    v.bar  = MagicMock(return_value="win")
    return v


def _model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _pl_module(model=None):
    if model is None:
        model = _model()
    pl_mod = MagicMock(spec=nn.Module)
    pl_mod.named_parameters = model.named_parameters
    pl_mod.parameters       = model.parameters
    return pl_mod, model


def _trainer(step=0):
    t = MagicMock()
    t.global_step = step
    return t


#  LightningHookProfiler 

@pytest.mark.lightning_logger
class TestLightningHookProfiler(unittest.TestCase):

    def test_empty_on_init(self):
        self.assertEqual(LightningHookProfiler().summary(), {})

    def test_records_timing(self):
        hp = LightningHookProfiler()
        hp.record_start()
        hp.record_end()
        self.assertEqual(hp.summary()["count"], 1)

    def test_multiple_records(self):
        hp = LightningHookProfiler()
        for _ in range(4):
            hp.record_start()
            hp.record_end()
        self.assertEqual(len(hp.overhead_ms), 4)

    def test_end_without_start_safe(self):
        hp = LightningHookProfiler()
        hp.record_end()
        self.assertEqual(hp.overhead_ms, [])

    def test_reset(self):
        hp = LightningHookProfiler()
        hp.record_start()
        hp.record_end()
        hp.reset()
        self.assertEqual(hp.overhead_ms, [])


# LightningOpProfiler 

@pytest.mark.lightning_logger
class TestLightningOpProfiler(unittest.TestCase):

    def test_summary_before_start(self):
        self.assertIn("not been started", LightningOpProfiler().summary())

    def test_export_before_start_raises(self):
        with self.assertRaises(RuntimeError):
            LightningOpProfiler().export_chrome_trace("x.json")

    def test_start_stop(self):
        p = LightningOpProfiler(wait=0, warmup=0, active=1)
        p.start()
        self.assertIsNotNone(p._profiler)
        p.stop()

    def test_step_before_start_safe(self):
        LightningOpProfiler().step()

# LightningGradientNormLogger lifecycle 

@pytest.mark.lightning_logger
@skip_no_lightning
class TestLightningLoggerLifecycle(unittest.TestCase):

    def setUp(self):
        self.vis = _vis()

    def test_no_hooks_before_train_start(self):
        cb = LightningGradientNormLogger(self.vis)
        self.assertEqual(len(cb._hooks), 0)

    def test_hooks_attached_on_train_start(self):
        cb = LightningGradientNormLogger(self.vis)
        pl_mod, _ = _pl_module()
        cb.on_train_start(_trainer(), pl_mod)
        self.assertEqual(len(cb._hooks), 4)  # 2 layers x (weight + bias)
        cb.on_train_end(_trainer(), pl_mod)

    def test_hooks_detached_on_train_end(self):
        cb = LightningGradientNormLogger(self.vis)
        pl_mod, _ = _pl_module()
        cb.on_train_start(_trainer(), pl_mod)
        cb.on_train_end(_trainer(), pl_mod)
        self.assertEqual(len(cb._hooks), 0)

    def test_double_train_start_warns(self):
        cb = LightningGradientNormLogger(self.vis) # Callback
        pl_mod, _ = _pl_module()
        cb.on_train_start(_trainer(), pl_mod)
        with self.assertWarns(UserWarning):
            cb.on_train_start(_trainer(), pl_mod)
        cb.on_train_end(_trainer(), pl_mod)


# LightningGradientNormLogger Visdom output

@pytest.mark.lightning_logger
@skip_no_lightning
class TestLightningLoggerVisdom(unittest.TestCase):

    def setUp(self):
        self.vis = _vis()
        pl_mod, self.model = _pl_module()
        self.pl_mod = pl_mod

    def _run(self, cb, n_steps):
        trainer = _trainer()
        cb.on_train_start(trainer, self.pl_mod)
        for i in range(n_steps):
            trainer.global_step = i
            self.model.zero_grad()
            out  = self.model(torch.randn(3, 4))
            loss = nn.CrossEntropyLoss()(out, torch.randint(0, 2, (3,)))
            loss.backward()
            cb.on_before_optimizer_step(trainer, self.pl_mod, MagicMock())
        cb.on_train_end(trainer, self.pl_mod)

    def test_vis_line_called(self):
        cb = LightningGradientNormLogger(self.vis, log_every=1)
        self._run(cb, 3)
        self.assertGreater(self.vis.line.call_count, 0)

    def test_log_every_throttles(self):
        cb = LightningGradientNormLogger(self.vis, log_every=3)
        self._run(cb, 9)
        self.assertEqual(self.vis.line.call_count, 3)

    def test_bar_disabled(self):
        cb = LightningGradientNormLogger(self.vis, log_every=1, log_per_layer=False)
        self._run(cb, 2)
        self.assertFalse(self.vis.bar.called)

    def test_custom_env(self):
        cb = LightningGradientNormLogger(self.vis, log_every=1, env="test_env")
        self._run(cb, 1)
        self.assertEqual(self.vis.line.call_args.kwargs["env"], "test_env")


# LightningGradientNormLogger profiler integration 

@pytest.mark.lightning_logger
@skip_no_lightning
class TestLightningLoggerProfilers(unittest.TestCase):

    def setUp(self):
        self.vis = _vis()
        pl_mod, self.model = _pl_module()
        self.pl_mod = pl_mod

    def test_hook_profiler_records(self):
        hp = LightningHookProfiler()
        cb = LightningGradientNormLogger(self.vis, hook_profiler=hp, log_every=999)
        t = _trainer()
        cb.on_train_start(t, self.pl_mod)
        self.model.zero_grad()
        out  = self.model(torch.randn(3, 4))
        loss = nn.CrossEntropyLoss()(out, torch.randint(0, 2, (3,)))
        loss.backward()
        self.assertGreater(len(hp.overhead_ms), 0)
        cb.on_train_end(t, self.pl_mod)

    def test_op_profiler_lifecycle(self):
        op = MagicMock(spec=LightningOpProfiler)
        cb = LightningGradientNormLogger(self.vis, op_profiler=op)
        t = _trainer()
        cb.on_train_start(t, self.pl_mod)
        op.start.assert_called_once()
        for i in range(3):
            t.global_step = i
            cb.on_before_optimizer_step(t, self.pl_mod, MagicMock())
        cb.on_train_end(t, self.pl_mod)
        self.assertEqual(op.step.call_count, 3)
        op.stop.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])