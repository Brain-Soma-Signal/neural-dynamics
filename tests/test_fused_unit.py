"""Tiny synthetic transaction guards only; no historical data or model imports."""
import copy
import unittest
import torch
from torch import nn
from neural_dynamics.fused_unit import FusedLinear


def same(a, b):
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


class FusedUnitGuards(unittest.TestCase):
    def make(self, dtype=torch.float32, bias=True):
        torch.manual_seed(46001)
        base = nn.Linear(2, 3, bias=bias, dtype=dtype)
        return base, FusedLinear(base)

    def tx(self, u, event, ids=('a',), mask=None, training=False):
        u.begin(ids, event, torch.ones(len(ids), dtype=torch.bool) if mask is None else mask, training=training)

    def test_identity_exact_and_copy_and_gate_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for bias in (True, False):
                base, u = self.make(dtype, bias)
                self.assertEqual(tuple(u.weight.shape), tuple(base.weight.shape))
                self.assertNotEqual(u.weight.data_ptr(), base.weight.data_ptr())
                self.assertTrue(all(p.requires_grad for p in u.parameters()))
                x = torch.tensor([[.3, -.2]], dtype=dtype)
                original = base.weight.detach().clone()
                for event in ('one', 'two'):
                    self.tx(u, event)
                    with torch.no_grad():
                        out = u(x)
                        self.assertEqual(out.numpy().tobytes(), base(x).numpy().tobytes())
                    receipts = u.commit()
                    u.apply_feedback(receipts, {'a': 1.})
                self.tx(u, 'three', training=True)
                u(x).sum().backward()
                self.assertIsNotNone(u.alpha.grad)
                self.assertIsNotNone(u.beta.grad)
                self.assertTrue(torch.isfinite(u.alpha.grad).all())
                self.assertGreater(u.alpha.grad.abs().sum().item(), 0)
                self.assertGreater(u.beta.grad.abs().sum().item(), 0)
                self.assertTrue(torch.equal(original, base.weight))
                u.rollback()

    def test_padding_repeat_prefix_readonly(self):
        _, u = self.make()
        with torch.no_grad():
            u.alpha.fill_(.2)
            u.beta.fill_(.15)
        x = torch.tensor([[[1., 2.], [3., 4.], [9., 9.]], [[2., 1.], [8., 8.], [7., 7.]]])
        mask = torch.tensor([[True, True, False], [True, False, False]])
        before = u.export_state()
        self.tx(u, 'repeat', ('a', 'b'), mask)
        out = u(x)
        self.assertTrue(same(before, dict(version=1, banks=u._banks, receipts=u._receipts)))
        changed = x.clone()
        changed[:, -1] += 30
        out2 = u(changed)
        self.assertTrue(torch.equal(out[:, :2], out2[:, :2]))
        prefix = u(changed[:, :2])
        self.assertTrue(torch.equal(out2[:, :2], prefix))
        self.assertTrue(torch.equal(out2, u(changed)))
        r = u.commit()
        linear = torch.nn.functional.linear(x, u.weight, u.bias)
        for j, i in enumerate(('a', 'b')):
            y = linear[j, (1, 0)[j]]
            phi = torch.tanh(y / y.square().mean().sqrt().clamp(min=1e-6))
            self.assertTrue(torch.equal(u._banks[i]['c'], .5 * phi))
            self.assertTrue(torch.equal(u._banks[i]['e'], .1 * phi))
        u.waive_feedback(r)

    def test_invalid_atomic_receipts_capacity_rollback(self):
        _, u = self.make()
        before = u.export_state()
        for ids, mask in [(('a', 'a'), torch.ones(2, dtype=torch.bool)),
                          (('a',), torch.zeros(1, dtype=torch.bool)),
                          (('a',), torch.ones(1)),
                          (tuple('abcde'), torch.ones(5, dtype=torch.bool))]:
            with self.assertRaises((ValueError, RuntimeError)):
                self.tx(u, 'bad', ids, mask)
            self.assertTrue(same(before, u.export_state()))
        with self.assertRaises(RuntimeError):
            u.apply_feedback({'a': dict(event_id='bad', consumed=False)}, {'a': 1.})
        self.tx(u, 'ok', ('a', 'b'))
        u(torch.ones(2, 2))
        with self.assertRaises(ValueError):
            u(torch.ones(2, 2, dtype=torch.float64))
        r = u.commit()  # invalid forward did not erase the valid pending summary
        before = u.export_state()
        with self.assertRaises(ValueError):
            u.apply_feedback(r, {'a': 1., 'b': 1.01})
        self.assertTrue(same(before, u.export_state()))
        with self.assertRaises(RuntimeError):
            self.tx(u, 'blocked')
        self.assertTrue(same(before, u.export_state()))
        u.apply_feedback(r, {'a': -1., 'b': .3})
        consumed = u.export_state()
        for method in (lambda: u.apply_feedback(r, {'a': 1., 'b': 1.}), lambda: u.waive_feedback(r)):
            with self.assertRaises(RuntimeError):
                method()
            self.assertTrue(same(consumed, u.export_state()))
        self.tx(u, 'rollback', ('c', 'd'))
        u(torch.zeros(2, 2))
        u.rollback()
        self.assertTrue(same(consumed, u.export_state()))
        self.tx(u, 'four', ('c', 'd'))
        u(torch.ones(2, 2))
        u.waive_feedback(u.commit())
        four = u.export_state()
        with self.assertRaises(ValueError):
            self.tx(u, 'overflow', ('e',))
        self.assertTrue(same(four, u.export_state()))

    def test_context_isolation_save_restore_reset_atomic(self):
        _, u = self.make()
        self.tx(u, 'initial', ('a', 'b'))
        u(torch.tensor([[1., 2.], [3., 4.]]))
        u.waive_feedback(u.commit())
        before = u.export_state()
        self.tx(u, 'a-only')
        u(torch.tensor([[5., 6.]]))
        receipts = u.commit()
        self.assertTrue(same(before['banks']['b'], u._banks['b']))
        saved = copy.deepcopy(u.state_dict())
        _, clone = self.make()
        clone.load_state_dict(saved)
        self.assertTrue(same(u.export_state(), clone.export_state()))
        clone.apply_feedback(receipts, {'a': .7})
        with self.assertRaises(RuntimeError):
            self.tx(clone, 'a-only')
        u.waive_feedback(receipts)
        before = u.export_state()
        bad = copy.deepcopy(before)
        bad['banks']['b']['w'][0] = float('nan')
        with self.assertRaises(ValueError):
            u.restore_state(bad)
        self.assertTrue(same(before, u.export_state()))
        u.clear_state()
        self.assertEqual(u.export_state()['banks'], {})
        u.restore_state(before)
        self.assertTrue(same(before, u.export_state()))

    def test_invalid_lifecycle_and_prefix_atomic(self):
        _, u = self.make()
        with self.assertRaises(RuntimeError):
            u(torch.ones(1, 2))
        self.tx(u, 'empty', mask=torch.tensor([[False, True]]))
        with self.assertRaises(RuntimeError):
            u.commit()
        with self.assertRaises(ValueError):
            u(torch.ones(1, 1, 2))
        self.assertIsNone(u._active['pending'])
        for action in (u.detach_state, u.clear_state, u.export_state):
            with self.assertRaises(RuntimeError):
                action()
        u(torch.ones(1, 2, 2))
        pending = u._active['pending'].clone()
        with self.assertRaises(ValueError):
            u(torch.full((1, 2, 2), float('nan')))
        self.assertTrue(torch.equal(pending, u._active['pending']))
        r = u.commit()
        state = u.export_state()
        with self.assertRaises(RuntimeError):
            u.commit()
        with self.assertRaises(ValueError):
            u.apply_feedback(r, {'a': float('nan')})
        self.assertTrue(same(state, u.export_state()))
        u.waive_feedback(r)
        u.restore_state(u.export_state())
        self.tx(u, 'again')
        with self.assertRaises(RuntimeError):
            u.begin(('b',), 'nested', torch.ones(1, dtype=torch.bool))
        u.rollback()

    def test_two_transaction_gradient_boundary(self):
        _, u = self.make(torch.float64)
        with torch.no_grad():
            u.beta.fill_(.15)
        x = torch.tensor([[.3, -.7]], dtype=torch.float64)
        self.tx(u, 't0', training=True)
        u(x)
        u.waive_feedback(u.commit())
        self.tx(u, 't1', training=True)
        u(x * 2).square().sum().backward()
        u.waive_feedback(u.commit())
        self.assertGreater(u.retention_logits.grad.abs().sum().item(), 0)
        self.assertIsNone(u._banks['a']['w'].grad_fn)
        self.assertIsNone(u._banks['a']['e'].grad_fn)
        before = u.export_state()
        with self.assertRaises(RuntimeError):
            self.tx(u, 'too-many', training=True)
        self.assertTrue(same(before, u.export_state()))
        u.detach_state()
        self.assertIsNone(u._banks['a']['c'].grad_fn)
        self.tx(u, 'inference')
        u(x)
        u.waive_feedback(u.commit())
        self.assertIsNone(u._banks['a']['c'].grad_fn)


if __name__ == '__main__':
    unittest.main()
