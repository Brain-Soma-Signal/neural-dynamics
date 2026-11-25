"""Stage53 artificial output cells with internal nonlinear dendritic branches.

Fixed disjoint columns reuse the inherited W. This is an engineering abstraction,
not a biological reconstruction, semantic memory, or scalar synaptogenesis.
Training banks retain m/h/v/a graphs across at most two committed events; p/e
never retain graphs. Call detach_state only after the paired backward and step.
"""
from __future__ import annotations
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from neural_dynamics.fused_unit import FusedLinear

SLOW_NAMES = ('branch_bias', 'memory_mix', 'slow_mix', 'gate_signal', 'gate_memory',
              'branch_retention_logits', 'slow_retention_logits', 'soma_retention_logits',
              'adaptation_retention_logits', 'output_gain', 'adaptation_strength')
RETAINED = ('m', 'h', 'v', 'a')
DEFAULT_OBSERVER = None


def cell_equations(x, weight, bias, parameters, state, branches):
    """Pure differentiable per-token equations; commits are deliberately separate."""
    base = F.linear(x, weight, bias)
    widths = [part.numel() for part in torch.tensor_split(torch.arange(weight.shape[1]), branches)]
    xs, ws = torch.split(x, widths, dim=-1), torch.split(weight, widths, dim=-1)
    u = torch.stack([F.linear(a, b) for a, b in zip(xs, ws)], dim=-1)
    q = u / torch.sqrt(u.square().mean(dim=-1, keepdim=True) + 1e-6)
    z = torch.tanh(q * (1 + torch.tanh(state['p'])) + parameters['memory_mix'] * state['m']
                   + parameters['slow_mix'] * state['h'])
    g = torch.softmax(parameters['branch_bias'] + parameters['gate_signal'] * z
                      + parameters['gate_memory'] * state['m'], dim=-1)
    r = (g * z).sum(dim=-1)
    s = torch.tanh(r + state['v'] - F.softplus(parameters['adaptation_strength']) * state['a'])
    if not torch.is_grad_enabled() and not torch.count_nonzero(parameters['output_gain']):
        out = base
    else:
        out = base + torch.tanh(parameters['output_gain']) * (s - base)
    return out, dict(base=base, u=u, q=q, z=z, g=g, r=r, s=s)


class DendriticLinear(FusedLinear):
    def __init__(self, linear: nn.Linear, *, branches=4, observer=None):
        nn.Module.__init__(self)
        if type(linear) is not nn.Linear or linear.weight.device.type != 'cpu' or linear.weight.dtype not in (torch.float32, torch.float64):
            raise ValueError('native CPU float32/float64 Linear required')
        if type(branches) is not int or not 1 <= branches <= linear.in_features:
            raise ValueError('nonempty balanced branch partitions required')
        self.in_features, self.out_features, self.branches = linear.in_features, linear.out_features, branches
        # Reuse the Parameter objects, avoiding even a transient full W clone.
        self.weight, self.bias = linear.weight, linear.bias
        for name in SLOW_NAMES:
            shape = (self.out_features, branches) if name in SLOW_NAMES[:7] else (self.out_features,)
            initial = .1 if name in ('memory_mix', 'slow_mix', 'gate_signal', 'gate_memory') else 0.
            if name in ('slow_retention_logits', 'adaptation_retention_logits'):
                initial = math.log(9)
            if name == 'adaptation_strength':
                initial = -2.
            self.register_parameter(name, nn.Parameter(self.weight.new_full(shape, initial)))
        self.observer = observer if observer is not None else DEFAULT_OBSERVER
        self._banks, self._receipts, self._active, self._training_transactions = {}, {}, None, 0

    def _empty(self):
        return {k: self.weight.new_zeros((self.out_features, self.branches) if k in ('m', 'h', 'p', 'e')
                                        else (self.out_features,)) for k in ('m', 'h', 'p', 'e', 'v', 'a')}

    def begin(self, context_ids, event_id, last_valid_mask, *, training=False):
        if self._active is not None:
            raise RuntimeError('transaction already active')
        if type(training) is not bool:
            raise ValueError('boolean training required')
        ids = tuple(context_ids)
        if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('unique nonempty string IDs required')
        if not isinstance(event_id, str) or not event_id:
            raise ValueError('nonempty event required')
        if len(set(self._banks) | set(ids)) > 4:
            raise ValueError('capacity four exceeded')
        for i in ids:
            receipt = self._receipts.get(i)
            if receipt and (not receipt['consumed'] or receipt['event_id'] == event_id):
                raise RuntimeError('consume receipt and use new event')
        mask = last_valid_mask
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.device.type != 'cpu' or mask.ndim not in (1, 2) or mask.shape[0] != len(ids) or not mask.reshape(len(ids), -1).any(dim=1).all():
            raise ValueError('nonempty CPU bool [batch] or [batch,T] mask required')
        if self._training_transactions >= 2 or (self._training_transactions and not training):
            raise RuntimeError('explicit detach required')
        snapshots = {i: {k: (v.clone() if training and k in RETAINED else v.detach().clone())
                         for k, v in self._banks.get(i, self._empty()).items()} for i in ids}
        self._active = dict(ids=ids, event_id=event_id, mask=mask.clone(), snapshots=snapshots,
                            training=training, pending=None)
        self._emit('begin')
        return self

    def forward(self, x):
        self._emit('call')
        tx = self._active
        if tx is None:
            raise RuntimeError('begin required')
        if x.ndim not in (2, 3) or x.shape[0] != len(tx['ids']) or x.shape[-1] != self.in_features or x.dtype != self.weight.dtype or x.device != self.weight.device or not torch.isfinite(x).all():
            raise ValueError('finite matching [batch,in] or [batch,T,in] input required')
        mask = tx['mask']
        if x.ndim == 2:
            if mask.ndim != 1:
                raise ValueError('2D input requires 1D mask')
        else:
            if mask.ndim != 2 or not 0 < x.shape[1] <= mask.shape[1]:
                raise ValueError('nonempty prefix required')
            mask = mask[:, :x.shape[1]]
            if not mask.any(dim=1).all():
                raise ValueError('each prefix must have a valid token')
        state = {k: torch.stack([tx['snapshots'][i][k] for i in tx['ids']]) for k in self._empty()}
        if x.ndim == 3:
            state = {k: v[:, None] for k, v in state.items()}
        out, details = cell_equations(x, self.weight, self.bias,
                                     {k: getattr(self, k) for k in SLOW_NAMES}, state, self.branches)
        if not all(torch.isfinite(v).all() for v in details.values()) or not torch.isfinite(out).all():
            raise ValueError('nonfinite cell output')
        indices = (torch.where(mask, torch.arange(x.shape[1]), -1).max(dim=1).values
                   if x.ndim == 3 else torch.zeros(x.shape[0], dtype=torch.long))
        pending = {k: (details[k][torch.arange(x.shape[0]), indices] if x.ndim == 3 else details[k])
                   for k in ('z', 'r', 's')}
        tx['pending'] = pending if tx['training'] else {k: v.detach() for k, v in pending.items()}
        if self.observer:
            self.observer('output', dict(unit=self, x=x, out=out, indices=indices, **details))
        return out

    def commit(self):
        tx = self._active
        if tx is None or tx['pending'] is None:
            raise RuntimeError('successful forward required')
        lm, lh, lv, la = [torch.sigmoid(getattr(self, k)) for k in
                          ('branch_retention_logits', 'slow_retention_logits', 'soma_retention_logits', 'adaptation_retention_logits')]
        updates = {}
        for row, i in enumerate(tx['ids']):
            old, pending = tx['snapshots'][i], tx['pending']
            m = lm * old['m'] + (1 - lm) * pending['z'][row]
            bank = dict(m=m, h=lh * old['h'] + (1 - lh) * m,
                        v=lv * old['v'] + (1 - lv) * pending['r'][row],
                        a=la * old['a'] + (1 - la) * pending['s'][row].square(),
                        e=(.9 * old['e'] + .1 * pending['z'][row]).detach(), p=old['p'].detach())
            updates[i] = bank if tx['training'] else {k: v.detach() for k, v in bank.items()}
        if not all(torch.isfinite(v).all() for b in updates.values() for v in b.values()):
            raise ValueError('nonfinite commit; no writes applied')
        receipts = {i: dict(event_id=tx['event_id'], consumed=False) for i in tx['ids']}
        self._banks.update(updates)
        self._receipts.update(receipts)
        self._active = None
        self._training_transactions += int(tx['training'])
        self._emit('commit')
        return deepcopy(receipts)

    def _consume(self, receipts, deltas):
        if self._active is not None:
            raise RuntimeError('feedback only between transactions')
        if not isinstance(receipts, dict) or not receipts:
            raise ValueError('nonempty receipts required')
        if deltas is not None and (not isinstance(deltas, dict) or set(deltas) != set(receipts)):
            raise ValueError('one public delta per receipt required')
        updates = {}
        for i, r in receipts.items():
            if i not in self._receipts or r != self._receipts[i] or r.get('consumed') is not False:
                raise RuntimeError('stale/consumed/unknown receipt')
            if deltas is not None:
                d = deltas[i]
                if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or not -1 <= d <= 1:
                    raise ValueError('finite public delta [-1,1] required')
                b = self._banks[i]
                updates[i] = (.995 * b['p'] + .01 * d * b['e']).clamp(-1, 1).detach()
        for i, p in updates.items():
            self._banks[i]['p'] = p
        for i in receipts:
            self._receipts[i] = dict(self._receipts[i], consumed=True)
        self._emit('waive' if deltas is None else 'feedback')

    def clear_state(self):
        if self._active is not None or self._training_transactions:
            raise RuntimeError('clear requires idle detached state')
        self._banks, self._receipts = {}, {}
        self._emit('clear')

    def export_state(self):
        state = super().export_state()
        state['version'] = 53
        return state

    def restore_state(self, state):
        if self._active is not None or self._training_transactions:
            raise RuntimeError('restore requires idle detached state')
        if not isinstance(state, dict) or set(state) != {'version', 'banks', 'receipts'} or state['version'] != 53:
            raise ValueError('invalid state envelope')
        banks, receipts = state['banks'], state['receipts']
        if not isinstance(banks, dict) or not isinstance(receipts, dict) or len(banks) > 4 or set(banks) != set(receipts):
            raise ValueError('invalid contexts')
        copied = {}
        empty = self._empty()
        for i, bank in banks.items():
            if not isinstance(i, str) or not i or not isinstance(bank, dict) or set(bank) != set(empty):
                raise ValueError('invalid bank')
            copied[i] = {}
            for k, v in bank.items():
                if not isinstance(v, torch.Tensor) or v.shape != empty[k].shape or v.dtype != self.weight.dtype or v.device != self.weight.device or not torch.isfinite(v).all() or (v.abs() > 1).any() or (k == 'a' and (v < 0).any()):
                    raise ValueError('invalid state tensor')
                copied[i][k] = v.detach().clone()
            r = receipts[i]
            if not isinstance(r, dict) or set(r) != {'event_id', 'consumed'} or not isinstance(r['event_id'], str) or not r['event_id'] or type(r['consumed']) is not bool:
                raise ValueError('invalid receipt')
        self._banks, self._receipts = copied, deepcopy(receipts)
        self._emit('restore')
