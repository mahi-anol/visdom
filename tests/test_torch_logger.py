import math
import sys
import unittest
from unittest.mock import MagicMock
import pytest
import torch
import torch.nn as nn
from visdom.loggers.base import compute_grad_norm, compute_layer_grad_norm
from visdom.loggers.torch_profiler import HookProfiler, TorchOpProfiler
from visdom.loggers.torch_logger import GradientNormLogger


def _vis():
    v = MagicMock()
    v.line = MagicMock(return_value="win")
    v.bar  = MagicMock(return_value="win")
    return v


def _model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _backward(model):
    out  = model(torch.randn(3, 4))
    loss = nn.CrossEntropyLoss()(out, torch.randint(0, 2, (3,)))
    loss.backward()


def _step(logger, model):
    model.zero_grad()
    _backward(model)
    logger.step()


# base.py 

@pytest.mark.torch_logger
class TestGradNormMath(unittest.TestCase):

    def test_no_grads_returns_zero(self):
        self.assertEqual(compute_grad_norm([torch.randn(3)]), 0.0)

    def test_l2_known_value(self):
        p = nn.Parameter(torch.ones(2, 2))
        p.grad = torch.ones(2, 2)
        self.assertAlmostEqual(compute_grad_norm([p], 2.0), 2.0, places=5)

    def test_multi_param(self):
        p1 = nn.Parameter(torch.zeros(2))
        p2 = nn.Parameter(torch.zeros(1))
        p1.grad = torch.tensor([3., 4.])
        p2.grad = torch.tensor([0.])
        self.assertAlmostEqual(compute_grad_norm([p1, p2]), 5.0, places=5)

    def test_layer_norm_l2(self):
        self.assertAlmostEqual(
            compute_layer_grad_norm(torch.tensor([3., 4.])), 5.0, places=5
        )

    def test_layer_norm_does_not_mutate(self):
        g = torch.tensor([1., 2.])
        orig = g.clone()
        compute_layer_grad_norm(g)
        self.assertTrue(torch.equal(g, orig))


# HookProfiler

@pytest.mark.torch_logger
class TestHookProfiler(unittest.TestCase):

    def test_empty_on_init(self):
        self.assertEqual(HookProfiler().summary(), {})

    def test_records_timing(self):
        hp = HookProfiler()
        hp.record_start()
        hp.record_end()
        self.assertEqual(hp.summary()["count"], 1)

    def test_multiple_records(self):
        hp = HookProfiler()
        for _ in range(5):
            hp.record_start()
            hp.record_end()
        self.assertEqual(len(hp.overhead_ms), 5)

    def test_end_without_start_safe(self):
        hp = HookProfiler()
        hp.record_end()
        self.assertEqual(hp.overhead_ms, [])

    def test_reset(self):
        hp = HookProfiler()
        hp.record_start()
        hp.record_end()
        hp.reset()
        self.assertEqual(hp.overhead_ms, [])


# orchOpProfiler 

@pytest.mark.torch_logger
class TestTorchOpProfiler(unittest.TestCase):

    def test_summary_before_run(self):
        self.assertIn("not been run", TorchOpProfiler().summary())

    def test_export_before_run_raises(self):
        with self.assertRaises(RuntimeError):
            TorchOpProfiler().export_chrome_trace("x.json")

    def test_context_manager(self):
        p = TorchOpProfiler(wait=0, warmup=0, active=1)
        with p:
            self.assertIsNotNone(p._profiler)

    def test_step_outside_context_safe(self):
        TorchOpProfiler().step()


# GradientNormLogger lifecycle 

@pytest.mark.torch_logger
class TestLoggerLifecycle(unittest.TestCase):

    def setUp(self):
        self.vis   = _vis()
        self.model = _model()

    def test_no_hooks_before_attach(self):
        logger = GradientNormLogger(self.vis, self.model)
        self.assertEqual(len(logger._hooks), 0)

    def test_attach_registers_hooks(self):
        logger = GradientNormLogger(self.vis, self.model)
        logger.attach()
        self.assertEqual(len(logger._hooks), 4)  # 2 layers x (weight + bias)
        logger.detach()

    def test_detach_clears_hooks(self):
        logger = GradientNormLogger(self.vis, self.model)
        logger.attach()
        logger.detach()
        self.assertEqual(len(logger._hooks), 0)

    def test_double_attach_warns(self):
        logger = GradientNormLogger(self.vis, self.model)
        logger.attach()
        with self.assertWarns(UserWarning):
            logger.attach()
        logger.detach()

    def test_context_manager(self):
        logger = GradientNormLogger(self.vis, self.model)
        with logger:
            self.assertGreater(len(logger._hooks), 0)
        self.assertEqual(len(logger._hooks), 0)

    def test_frozen_param_skipped(self):
        model = nn.Linear(4, 2)
        model.bias.requires_grad = False
        logger = GradientNormLogger(self.vis, model)
        logger.attach()
        self.assertEqual(len(logger._hooks), 1)
        logger.detach()


# GradientNormLogger Visdom output 

@pytest.mark.torch_logger
class TestLoggerVisdom(unittest.TestCase):

    def setUp(self):
        self.vis   = _vis()
        self.model = _model()

    def test_vis_line_called(self):
        logger = GradientNormLogger(self.vis, self.model, log_every=1)
        with logger:
            _step(logger, self.model)
        self.assertTrue(self.vis.line.called)

    def test_no_vis_call_before_log_every(self):
        logger = GradientNormLogger(self.vis, self.model, log_every=5)
        with logger:
            _step(logger, self.model)
        self.assertFalse(self.vis.line.called)

    def test_log_every_throttles(self):
        logger = GradientNormLogger(self.vis, self.model, log_every=3)
        with logger:
            for _ in range(9):
                _step(logger, self.model)
        self.assertEqual(self.vis.line.call_count, 3)

    def test_bar_disabled(self):
        logger = GradientNormLogger(
            self.vis, self.model, log_every=1, log_per_layer=False
        )
        with logger:
            _step(logger, self.model)
        self.assertFalse(self.vis.bar.called)

    def test_custom_env(self):
        logger = GradientNormLogger(self.vis, self.model, log_every=1, env="exp1")
        with logger:
            _step(logger, self.model)
        self.assertEqual(self.vis.line.call_args.kwargs["env"], "exp1")

    def test_norms_are_finite(self):
        norms = []
        self.vis.line.side_effect = lambda **kw: norms.append(kw["Y"][0])
        logger = GradientNormLogger(self.vis, self.model, log_every=1)
        opt = torch.optim.Adam(self.model.parameters())
        with logger:
            for _ in range(5):
                opt.zero_grad()
                _backward(self.model)
                logger.step()
                opt.step()
        self.assertTrue(all(math.isfinite(n) for n in norms))


# GradientNormLogger profiler integration 

@pytest.mark.torch_logger
class TestLoggerProfilers(unittest.TestCase):

    def setUp(self):
        self.vis   = _vis()
        self.model = _model()

    def test_hook_profiler_records(self):
        hp = HookProfiler()
        logger = GradientNormLogger(self.vis, self.model, log_every=999, hook_profiler=hp)
        with logger:
            _backward(self.model)
        self.assertGreater(len(hp.overhead_ms), 0)

    def test_hook_profiler_count_matches(self):
        hp = HookProfiler()
        n_params = sum(1 for p in self.model.parameters() if p.requires_grad)
        logger = GradientNormLogger(self.vis, self.model, log_every=999, hook_profiler=hp)
        with logger:
            for _ in range(3):
                _step(logger, self.model)
        self.assertEqual(len(hp.overhead_ms), n_params * 3)

    def test_op_profiler_step_called(self):
        op = MagicMock(spec=TorchOpProfiler)
        logger = GradientNormLogger(self.vis, self.model, log_every=1, op_profiler=op)
        with logger:
            for _ in range(3):
                _step(logger, self.model)
        self.assertEqual(op.step.call_count, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])