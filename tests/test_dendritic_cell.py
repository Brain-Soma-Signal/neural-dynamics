"""Alternate-only stage53 guards: never execute the frozen formal tiny cases."""
import copy
import io
import unittest
import numpy as np
import torch
from torch import nn
from neural_dynamics.dendritic_cell import DendriticLinear, SLOW_NAMES, cell_equations
from neural_dynamics.dendritic_qwen import DendriticQwenController, migrate_projections
from neural_dynamics.fused_qwen import tied_parameter_groups

COUNTS = dict(forwards=0, backwards=0, optimizer_updates=0, model_loads=0, formal_cases=0)
RETENTIONS = ('branch_retention_logits', 'slow_retention_logits', 'soma_retention_logits', 'adaptation_retention_logits')


def observed(event, value):
    if event == 'call':
        COUNTS['forwards'] += 1


def make(dtype=torch.float32):
    torch.manual_seed(9829999)
    u = DendriticLinear(nn.Linear(6, 3, dtype=dtype), branches=2, observer=observed)
    with torch.no_grad():
        u.output_gain.fill_(.2)
    return u


def event(u, x, name, *, training=False, feedback=None):
    u.begin(['c'], name, torch.ones(x.shape[:2], dtype=torch.bool), training=training)
    out = u(x)
    receipts = u.commit()
    if feedback is None:
        u.waive_feedback(receipts)
    else:
        u.apply_feedback(receipts, {'c': feedback})
    return out


def numpy_event(u, x, state):
    """Independent NumPy implementation includes state and feedback equations."""
    p = {k: getattr(u, k).detach().numpy() for k in SLOW_NAMES}
    w, bias = u.weight.detach().numpy(), u.bias.detach().numpy()
    parts = np.array_split(np.arange(x.shape[-1]), u.branches)
    contributions = np.stack([x[..., ids] @ w[:, ids].T for ids in parts], axis=-1)
    q = contributions / np.sqrt(np.mean(contributions ** 2, axis=-1, keepdims=True) + 1e-6)
    z = np.tanh(q * (1 + np.tanh(state['p'])) + p['memory_mix'] * state['m'] + p['slow_mix'] * state['h'])
    scores = p['branch_bias'] + p['gate_signal'] * z + p['gate_memory'] * state['m']
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    gate = exp / exp.sum(axis=-1, keepdims=True)
    r = (gate * z).sum(axis=-1)
    s = np.tanh(r + state['v'] - np.logaddexp(0, p['adaptation_strength']) * state['a'])
    base = x @ w.T + bias
    out = base + np.tanh(p['output_gain']) * (s - base)
    sig = lambda key: 1 / (1 + np.exp(-p[key]))
    m = sig(RETENTIONS[0]) * state['m'] + (1 - sig(RETENTIONS[0])) * z[0, -1]
    new = dict(m=m, h=sig(RETENTIONS[1]) * state['h'] + (1 - sig(RETENTIONS[1])) * m,
               v=sig(RETENTIONS[2]) * state['v'] + (1 - sig(RETENTIONS[2])) * r[0, -1],
               a=sig(RETENTIONS[3]) * state['a'] + (1 - sig(RETENTIONS[3])) * s[0, -1] ** 2,
               e=.9 * state['e'] + .1 * z[0, -1], p=state['p'].copy())
    return out, new


class DendriticGuards(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = torch.linspace(-.63, .77, 18).reshape(1, 3, 6)

    def test_numpy_two_events_and_feedback(self):
        u = make()
        state = {k: v.numpy() for k, v in u._empty().items()}
        with torch.no_grad():
            for i, x in enumerate((self.x, self.x.flip(-1) * .57)):
                expected, state = numpy_event(u, x.numpy(), state)
                actual = event(u, x, str(i), feedback=1 if i == 0 else None)
                if i == 0:
                    state['p'] = np.clip(.995 * state['p'] + .01 * state['e'], -1, 1)
                np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6, rtol=1e-5)
                for k, v in state.items():
                    np.testing.assert_allclose(u._banks['c'][k].numpy(), v, atol=1e-6, rtol=1e-5)

    def test_identity_history_and_branch_gate_lesion(self):
        u = make()
        seen = []
        def observe(e, d):
            observed(e, d)
            if e == 'output':
                seen.append(d)
        u.observer = observe
        with torch.no_grad():
            u.output_gain.zero_()
            for i in range(2):
                y = event(u, self.x, str(i), feedback=1)
                self.assertTrue(torch.equal(y.view(torch.uint8), torch.nn.functional.linear(self.x, u.weight, u.bias).view(torch.uint8)))
            u.output_gain.fill_(.2)
            u.begin(['c'], 'a', torch.ones(1, 3, dtype=torch.bool))
            normal = u(self.x)
            u.rollback()
            lesion = self.x.clone(); lesion[..., :3] = 0
            u.begin(['c'], 'b', torch.ones(1, 3, dtype=torch.bool))
            altered = u(lesion); u.rollback()
        self.assertFalse(torch.equal(normal, altered))
        self.assertFalse(torch.equal(seen[-2]['s'], seen[-1]['s']))
        self.assertTrue(torch.equal(seen[-1]['u'][..., 0], torch.zeros_like(seen[-1]['u'][..., 0])))
        self.assertTrue(torch.allclose(seen[-1]['g'].sum(-1), torch.ones(1, 3, 3)))
        self.assertEqual(sum(getattr(u, k).numel() for k in SLOW_NAMES), 3 * (7 * 2 + 4))

    def test_causality_prefix_overwrite_last_valid_rollback(self):
        u = make()
        mask = torch.tensor([[True, True, False]])
        with torch.no_grad():
            u.begin(['c'], 'prefix', mask)
            full = u(self.x)
            pending = {k: v.clone() for k, v in u._active['pending'].items()}
            future = self.x.clone(); future[:, -1] += 20
            later = u(future)
            self.assertTrue(torch.equal(full[:, :2], later[:, :2]))
            u(self.x[:, :2])
            for k in pending:
                self.assertTrue(torch.equal(pending[k], u._active['pending'][k]))
            u.rollback()
        self.assertEqual(u._banks, {})

    def test_feedback_zero_vs_waive_receipts_capacity(self):
        u = make()
        with torch.no_grad():
            event(u, self.x, 'a', feedback=1)
            checkpoint = u.export_state()
            event(u, self.x, 'b', feedback=0)
            zero = u._banks['c']['p'].clone()
            u.restore_state(checkpoint)
            event(u, self.x, 'b')
            self.assertTrue(torch.equal(zero, .995 * u._banks['c']['p']))
            for i in range(3):
                u.begin([str(i)], 'c', torch.ones(1, 3, dtype=torch.bool)); u(self.x)
                r = u.commit(); u.waive_feedback(r)
                with self.assertRaises(RuntimeError): u.waive_feedback(r)
            with self.assertRaises(ValueError): u.begin(['overflow'], 'c', torch.ones(1, 3, dtype=torch.bool))

    def test_restore_validation_atomic_and_checkpoint(self):
        u = make()
        with torch.no_grad(): event(u, self.x, 'a', feedback=1)
        saved = u.export_state()
        for bad in ('shape', 'finite', 'negative_adaptation', 'schema'):
            corrupt = copy.deepcopy(saved)
            if bad == 'shape': corrupt['banks']['c']['m'] = torch.zeros(1)
            if bad == 'finite': corrupt['banks']['c']['v'][0] = float('nan')
            if bad == 'negative_adaptation': corrupt['banks']['c']['a'][0] = -.1
            if bad == 'schema': corrupt['banks']['c']['hidden_label'] = torch.zeros(3)
            with self.assertRaises(ValueError): u.restore_state(corrupt)
            for k in saved['banks']['c']: self.assertTrue(torch.equal(saved['banks']['c'][k], u._banks['c'][k]))
        buffer = io.BytesIO(); torch.save(u.state_dict(), buffer); buffer.seek(0)
        rebuilt = make(); rebuilt.load_state_dict(torch.load(buffer, weights_only=False))
        with torch.no_grad():
            a = event(u, self.x, 'b'); b = event(rebuilt, self.x, 'b')
        self.assertTrue(torch.equal(a, b))
        for k in u._banks['c']: self.assertTrue(torch.equal(u._banks['c'][k], rebuilt._banks['c'][k]))

    def test_retention_fp64_finite_difference(self):
        # Different alternate size/seed from both locked tiny cases.
        torch.manual_seed(9830003)
        u = DendriticLinear(nn.Linear(5, 2, dtype=torch.float64), branches=3, observer=observed)
        with torch.no_grad(): u.output_gain.fill_(.23)
        x = torch.linspace(-.82, .71, 10, dtype=torch.float64).reshape(1, 2, 5)
        def loss():
            u.detach_state(); u.clear_state()
            event(u, x, 'o', training=True, feedback=1)
            return event(u, x.flip(-1) * .61, 'q', training=True)[0, -1].square().sum()
        value = loss(); COUNTS['backwards'] += 1; value.backward()
        analytic = {k: getattr(u, k).grad.clone() for k in RETENTIONS}
        for name in RETENTIONS:
            p = getattr(u, name)
            self.assertGreater(float(analytic[name].norm()), 0)
            for index in range(p.numel()):
                with torch.no_grad(): p.reshape(-1)[index] += 1e-5
                plus = float(loss().detach())
                with torch.no_grad(): p.reshape(-1)[index] -= 2e-5
                minus = float(loss().detach())
                with torch.no_grad(): p.reshape(-1)[index] += 1e-5
                self.assertAlmostEqual(float(analytic[name].reshape(-1)[index]), (plus - minus) / 2e-5, delta=2e-9)
        u.detach_state()

    def test_two_event_credit_detach_control_and_sgd(self):
        norms = []
        for detached in (False, True):
            u = make()
            old = {k: p.detach().clone() for k, p in u.named_parameters()}
            event(u, self.x, 'o', training=True, feedback=1)
            for k in ('p', 'e'): self.assertFalse(u._banks['c'][k].requires_grad)
            if detached: u.detach_state()
            result = event(u, self.x.flip(-1) * .57, 'q', training=True)
            loss = result[0, -1].square().mean(); COUNTS['backwards'] += 1; loss.backward()
            norms.append({k: 0. if getattr(u, k).grad is None else float(getattr(u, k).grad.norm()) for k in RETENTIONS})
            if not detached:
                for k in SLOW_NAMES:
                    self.assertIsNotNone(getattr(u, k).grad, k)
                    self.assertGreater(float(getattr(u, k).grad.norm()), 0, k)
                opt = torch.optim.SGD(u.parameters(), lr=.01)
                torch.nn.utils.clip_grad_norm_(u.parameters(), 1)
                COUNTS['optimizer_updates'] += 1; opt.step(); opt.zero_grad(set_to_none=True)
                for k, p in u.named_parameters(): self.assertFalse(torch.equal(old[k], p), k)
            u.detach_state()
        self.assertTrue(all(v > 0 for v in norms[0].values()))
        self.assertEqual(set(norms[1].values()), {0.})

    def test_retained_lifecycle_and_nonleaf_atomic_recovery(self):
        first, second = make(), make()
        controller = DendriticQwenController({'first': first, 'second': second})
        controller.begin(['c'], 'o', torch.ones(1, 3, dtype=torch.bool), training=True)
        first(self.x); second(self.x)
        r = controller.commit(); controller.waive_feedback(r)
        original = first._banks['c']['m']
        self.assertIsNotNone(original.grad_fn)
        bad = controller.export_state(); bad['second']['banks']['c']['m'] = torch.zeros(1)
        # detach restrictions are tested separately, then late failure during begin
        for operation in (controller.clear_state, lambda: controller.restore_state(bad), lambda: controller.set_gains(.1), lambda: controller.begin(['c'], 'inference', torch.ones(1, 3, dtype=torch.bool))):
            with self.assertRaises(RuntimeError): operation()
        second._receipts['c']['consumed'] = False
        with self.assertRaises(RuntimeError): controller.begin(['c'], 'q', torch.ones(1, 3, dtype=torch.bool), training=True)
        self.assertIs(first._banks['c']['m'], original)
        self.assertIsNone(first._active); self.assertIsNone(controller.active)
        second._receipts['c']['consumed'] = True
        controller.begin(['c'], 'q', torch.ones(1, 3, dtype=torch.bool), training=True)
        first(self.x); second(self.x)
        pending = second._active['pending']
        second._active['pending'] = None
        with self.assertRaises(RuntimeError): controller.commit()
        self.assertIs(first._banks['c']['m'], original)
        self.assertIsNotNone(first._active)
        self.assertIsNotNone(controller.active)
        second._active['pending'] = pending
        r = controller.commit(); controller.waive_feedback(r)
        with self.assertRaises(RuntimeError): controller.begin(['c'], 'third', torch.ones(1, 3, dtype=torch.bool), training=True)
        controller.detach_state()
        state = controller.export_state(); old_bank = first._banks['c']['m']
        state['second']['banks']['c']['m'] = torch.zeros(1)
        with self.assertRaises(ValueError): controller.restore_state(state)
        self.assertIs(first._banks['c']['m'], old_bank)

    def test_migration_shapes_parameters_ties_and_prevalidation(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__(); self.a = nn.Linear(6, 3); self.b = nn.Linear(3, 2)
                self.emb = nn.Embedding(4, 2); self.head = nn.Linear(2, 4, bias=False); self.head.weight = self.emb.weight
        model = Model()
        before = {n: (p, p.detach().clone()) for n, p in model.named_parameters(remove_duplicate=False)}
        ties = tied_parameter_groups(model)
        inventory = {'targets': [{'name':'a', 'weight_shape':[3, 6]}, {'name':'b', 'weight_shape':[9, 3]}]}
        with self.assertRaises(ValueError): migrate_projections(model, inventory, branches=2)
        self.assertIs(type(model.a), nn.Linear)
        inventory['targets'][1]['weight_shape'] = [2, 3]
        c = migrate_projections(model, inventory, branches=2)
        self.assertEqual(len(c.units), 2)
        self.assertEqual(ties, tied_parameter_groups(model))
        for n, p in model.named_parameters(remove_duplicate=False):
            if n in before:
                self.assertIs(p, before[n][0]); self.assertTrue(torch.equal(p, before[n][1]))
        with self.assertRaises(TypeError): DendriticQwenController({'bad': nn.Linear(2, 2)})


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DendriticGuards))
    print('ACTUAL_ALTERNATE_GUARD_COUNTS', COUNTS)
    raise SystemExit(0 if result.wasSuccessful() else 1)
