"""Tiny two-operator migration/controller guards, without real Qwen or optimizer."""
import unittest
from unittest.mock import patch
import torch
from torch import nn
from neural_dynamics.fused_qwen import (migrate_projections, parameter_fingerprints,
                               tied_parameter_groups, reference_copy)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 3)
        self.b = nn.Linear(3, 2)
        self.embed = nn.Embedding(3, 2)
        self.head = nn.Linear(2, 3, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x):
        return self.b(torch.tanh(self.a(x)))


class FusedQwenGuards(unittest.TestCase):
    def make(self):
        torch.manual_seed(47001)
        model = Tiny()
        before = parameter_fingerprints(model)
        ties = tied_parameter_groups(model)
        ctl = migrate_projections(model, dict(targets=[dict(name='a', weight_shape=[3, 2]), dict(name='b', weight_shape=[2, 3])]))
        self.assertEqual(before, parameter_fingerprints(model, before))
        self.assertEqual(ties, tied_parameter_groups(model))
        return model, ctl

    def begin(self, ctl, event='event', training=False):
        ctl.begin(('tiny',), event, torch.ones(1, 2, dtype=torch.bool), training=training)

    def assert_refs(self, ctl, before):
        after = ctl._checkpoint_references()
        self.assertTrue(equal(before, after))
        for n in before:
            for i, bank in before[n]['banks'].items():
                for k, value in bank.items():
                    self.assertIs(value, after[n]['banks'][i][k])
            if before[n]['active'] is not None:
                self.assertIs(before[n]['active']['pending'], after[n]['active']['pending'])

    def late_failure(self, obj, method):
        original = getattr(obj, method)
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('injected after late-unit mutation')
        return patch.object(obj, method, fail)

    def test_migration_zero_identity_and_trainability(self):
        torch.manual_seed(47001)
        model = Tiny()
        x = torch.tensor([[[.1, -.3], [.4, .8]]])
        with torch.no_grad():
            baseline = model(x)
        before = parameter_fingerprints(model)
        ctl = migrate_projections(model, dict(targets=[dict(name='a', weight_shape=[3, 2]), dict(name='b', weight_shape=[2, 3])]))
        self.begin(ctl)
        with torch.no_grad():
            actual = model(x)
        self.assertEqual(actual.numpy().tobytes(), baseline.numpy().tobytes())
        receipts = ctl.commit()
        ctl.apply_feedback(receipts, {'tiny': .5})
        self.assertEqual(before, parameter_fingerprints(model, before))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        state = ctl.export_state()
        ctl.clear_state()
        ctl.restore_state(state)
        self.assertTrue(equal(state, ctl.export_state()))

    def test_atomic_late_begin_commit_feedback_and_restore(self):
        model, ctl = self.make()
        late = ctl.units['b']
        initial = ctl._checkpoint_references()
        with self.late_failure(late, 'begin'):
            with self.assertRaises(RuntimeError):
                self.begin(ctl)
        self.assert_refs(ctl, initial)
        self.assertIsNone(ctl.active)
        self.begin(ctl)
        model(torch.ones(1, 2, 2))
        pending = ctl._checkpoint_references()
        with self.late_failure(late, 'commit'):
            with self.assertRaises(RuntimeError):
                ctl.commit()
        self.assert_refs(ctl, pending)
        self.assertIsNotNone(ctl.active)
        r = ctl.commit()
        committed = ctl._checkpoint_references()
        with self.late_failure(late, 'apply_feedback'):
            with self.assertRaises(RuntimeError):
                ctl.apply_feedback(r, {'tiny': .5})
        self.assert_refs(ctl, committed)
        ctl.apply_feedback(r, {'tiny': .5})
        good = ctl.export_state()
        malformed = reference_copy(good)
        malformed['b'] = dict(malformed['b'], version=-1)
        before = ctl._checkpoint_references()
        with self.assertRaises(ValueError):
            ctl.restore_state(malformed)
        self.assert_refs(ctl, before)
        with self.assertRaises(RuntimeError):
            ctl.apply_feedback(r, {'tiny': .5})
        self.assert_refs(ctl, before)

    def test_graph_references_survive_failed_second_commit(self):
        model, ctl = self.make()
        ctl.set_gains(.1, .02)
        x = torch.tensor([[[.1, -.2], [.3, .4]]])
        self.begin(ctl, 'first', training=True)
        model(x)
        ctl.apply_feedback(ctl.commit(), {'tiny': .5})
        connected = {n: u._banks['tiny']['c'] for n, u in ctl.units.items()}
        self.assertTrue(all(t.grad_fn is not None for t in connected.values()))
        self.begin(ctl, 'second', training=True)
        out = model(x * 1.3)
        saved = ctl._checkpoint_references()
        with self.late_failure(ctl.units['b'], 'commit'):
            with self.assertRaises(RuntimeError):
                ctl.commit()
        self.assert_refs(ctl, saved)
        for n, t in connected.items():
            self.assertIs(t, ctl.units[n]._banks['tiny']['c'])
        ctl.waive_feedback(ctl.commit())
        out.square().sum().backward()
        for u in ctl.units.values():
            self.assertTrue(torch.isfinite(u.retention_logits.grad).all())
            self.assertGreater(float(u.retention_logits.grad.norm()), 0)
        ctl.detach_state()
        self.assertTrue(all(u._banks['tiny']['c'].grad_fn is None for u in ctl.units.values()))

    def test_invalid_inventory_prevalidated_without_mutation(self):
        torch.manual_seed(47001)
        model = Tiny()
        a, b = model.a, model.b
        bad = dict(targets=[dict(name='a', weight_shape=[3, 2]), dict(name='b', weight_shape=[999, 3])])
        with self.assertRaises(ValueError):
            migrate_projections(model, bad)
        self.assertIs(model.a, a)
        self.assertIs(model.b, b)


    def test_missing_coverage_and_rollback_are_atomic(self):
        model, ctl = self.make()
        self.begin(ctl)
        model.a(torch.ones(1, 2, 2))
        before = ctl._checkpoint_references()
        with self.assertRaises(RuntimeError):
            ctl.commit()
        self.assert_refs(ctl, before)
        with self.late_failure(ctl.units['b'], 'rollback'):
            with self.assertRaises(RuntimeError):
                ctl.rollback()
        self.assert_refs(ctl, before)
        ctl.rollback()
        self.assertEqual(ctl.export_state()['a']['banks'], {})
        self.assertEqual(ctl.counts['commit'], 0)
        self.assertEqual(ctl.counts['rollback'], 1)


if __name__ == '__main__':
    unittest.main()
