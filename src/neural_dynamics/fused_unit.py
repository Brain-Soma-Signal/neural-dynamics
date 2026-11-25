"""Stage46 reusable inherited-plus-state linear operator (CPU float32/float64).

One active batch transaction freezes its banks. Forward only stages the most
recent base-linear last valid token, never accumulated; commit is explicit.
Training graph retention is opt-in and bounded to two transactions between
``detach_state`` calls (call after every optimizer step). w/e never retain graphs.
Banks/receipts are saved via get_extra_state/state_dict or export_state.
"""
from __future__ import annotations
from copy import deepcopy
from collections.abc import Callable, Sequence
import torch
from torch import nn
from torch.nn import functional as F

# Runner installs a formal-only audit hook; ordinary guards have a separate ledger.
DEFAULT_OBSERVER = None


def fused_output(base_y, alpha, beta, c_snapshot, w_snapshot):
    """Differentiable pure equation shared by transactions and FP64 checks."""
    return base_y * (1 + torch.tanh(alpha) * torch.tanh(w_snapshot)) + torch.tanh(beta) * c_snapshot


class FusedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, *, observer: Callable | None = None):
        super().__init__()
        if linear.weight.device.type != 'cpu' or linear.weight.dtype not in (torch.float32, torch.float64):
            raise ValueError('CPU float32/float64 required')
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone())
        for name in ('alpha', 'beta', 'retention_logits'):
            self.register_parameter(name, nn.Parameter(self.weight.new_zeros(self.out_features)))
        self.observer = observer if observer is not None else DEFAULT_OBSERVER
        self._banks, self._receipts = {}, {}
        self._active = None
        self._training_transactions = 0

    def _zero(self):
        return self.weight.new_zeros(self.out_features)

    def begin(self, context_ids: Sequence[str], event_id: str, last_valid_mask: torch.Tensor,
              *, training: bool = False):
        if self._active is not None:
            raise RuntimeError('transaction already active')
        ids = tuple(context_ids)
        if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('unique nonempty string context IDs required')
        if not isinstance(event_id, str) or not event_id:
            raise ValueError('nonempty event ID required')
        if len(set(self._banks) | set(ids)) > 4:
            raise ValueError('capacity is four contexts; clear explicitly')
        for i in ids:
            receipt = self._receipts.get(i)
            if receipt and (not receipt['consumed'] or receipt['event_id'] == event_id):
                raise RuntimeError('prior receipt must be consumed and event must be new')
        mask = last_valid_mask
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.device.type != 'cpu':
            raise ValueError('CPU boolean mask required')
        if mask.ndim not in (1, 2) or mask.shape[0] != len(ids) or (mask.ndim == 2 and mask.shape[1] == 0):
            raise ValueError('mask must be [B] or [B,T]')
        if not mask.reshape(len(ids), -1).any(dim=1).all():
            raise ValueError('each context must have a valid token')
        if training and self._training_transactions >= 2:
            raise RuntimeError('detach_state required after two training transactions')
        if not training and self._training_transactions:
            raise RuntimeError('detach_state required before inference')
        snapshots = {}
        for i in ids:
            bank = self._banks.get(i, {k: self._zero() for k in ('c', 'w', 'e')})
            snapshots[i] = {k: (v.clone() if training and k == 'c' else v.detach().clone()) for k, v in bank.items()}
        # Validation above is complete: begin does not allocate committed contexts.
        self._active = dict(ids=ids, event_id=event_id, mask=mask.clone(), snapshots=snapshots,
                            training=bool(training), pending=None)
        self._emit('begin')
        return self

    def _emit(self, event):
        if self.observer is not None:
            self.observer(event, self)

    def forward(self, x: torch.Tensor):
        if self.observer is not None:
            self.observer('call', self)
        tx = self._active
        if tx is None:
            raise RuntimeError('begin required')
        if x.ndim not in (2, 3) or x.shape[0] != len(tx['ids']) or x.shape[-1] != self.in_features:
            raise ValueError('input must be [B,in] or [B,T,in]')
        if x.device != self.weight.device or x.dtype != self.weight.dtype or not torch.isfinite(x).all():
            raise ValueError('input dtype/device/finite values must match weight')
        mask = tx['mask']
        if x.ndim == 2:
            if mask.ndim != 1:
                raise ValueError('2D input requires [B] mask')
        else:
            if mask.ndim != 2 or not 0 < x.shape[1] <= mask.shape[1]:
                raise ValueError('3D input requires [B,Tmax] mask and nonempty prefix T<=Tmax')
            mask = mask[:, :x.shape[1]]
            if not mask.any(dim=1).all():
                raise ValueError('every supplied prefix row must contain a valid token')
        y = F.linear(x, self.weight, self.bias)
        c = torch.stack([tx['snapshots'][i]['c'] for i in tx['ids']])
        w = torch.stack([tx['snapshots'][i]['w'] for i in tx['ids']])
        if x.ndim == 3:
            c, w = c[:, None, :], w[:, None, :]
        # The inference identity branch preserves bytes (including signed zero).
        # Training always keeps gates in the graph, including their zero gradients.
        if not torch.is_grad_enabled() and not torch.count_nonzero(self.alpha) and not torch.count_nonzero(self.beta):
            out = y
        else:
            out = fused_output(y, self.alpha, self.beta, c, w)
        if not torch.isfinite(out).all() or not torch.isfinite(y).all():
            raise ValueError('nonfinite output')
        if x.ndim == 2:
            pending = y
            indices = torch.zeros(x.shape[0], dtype=torch.long)
        else:
            positions = torch.arange(x.shape[1]).expand(x.shape[0], -1)
            indices = torch.where(mask, positions, -1).max(dim=1).values
            pending = y[torch.arange(x.shape[0]), indices]
        tx['pending'] = pending if tx['training'] else pending.detach()
        if self.observer is not None:
            self.observer('output', dict(unit=self, x=x, y=y, out=out, pending=pending, indices=indices,
                                        context_ids=tx['ids'], event_id=tx['event_id']))
        return out

    def commit(self):
        tx = self._active
        if tx is None or tx['pending'] is None:
            raise RuntimeError('active transaction with successful forward required')
        y = tx['pending']
        phi = torch.tanh(y / torch.sqrt(torch.mean(y.square(), dim=-1, keepdim=True)).clamp(min=1e-6))
        retention = torch.sigmoid(self.retention_logits)
        updates = {}
        for row, i in enumerate(tx['ids']):
            old = tx['snapshots'][i]
            c = retention * old['c'] + (1 - retention) * phi[row]
            updates[i] = dict(c=c if tx['training'] else c.detach(),
                              e=(.9 * old['e'] + .1 * phi[row]).detach(), w=old['w'].detach())
        if not all(torch.isfinite(v).all() for bank in updates.values() for v in bank.values()):
            raise ValueError('nonfinite commit; no writes applied')
        receipts = {i: dict(event_id=tx['event_id'], consumed=False) for i in tx['ids']}
        self._banks.update(updates)
        self._receipts.update(receipts)
        self._training_transactions += int(tx['training'])
        self._active = None
        if self.observer is not None:
            self.observer('commit', dict(unit=self, phi=phi, pending=y, receipts_output=receipts))
        return deepcopy(receipts)

    def apply_feedback(self, receipts: dict, deltas: dict):
        self._consume(receipts, deltas)

    def waive_feedback(self, receipts: dict):
        self._consume(receipts, None)

    def _consume(self, receipts, deltas):
        if self._active is not None:
            raise RuntimeError('feedback only between transactions')
        if not isinstance(receipts, dict) or not receipts:
            raise ValueError('nonempty receipts required')
        if deltas is not None and (not isinstance(deltas, dict) or set(deltas) != set(receipts)):
            raise ValueError('one scalar delta for every supplied receipt required')
        updates = {}
        for i, receipt in receipts.items():
            if i not in self._receipts or receipt != self._receipts[i] or receipt.get('consumed') is not False:
                raise RuntimeError('stale, consumed, or unknown receipt')
            if deltas is not None:
                delta = deltas[i]
                if isinstance(delta, bool) or not isinstance(delta, (float, int)) or not -1 <= delta <= 1:
                    raise ValueError('finite public delta in [-1,1] required')
                bank = self._banks[i]
                updates[i] = (.99 * bank['w'] + .001 * delta * bank['e']).clamp(-1, 1).detach()
        for i, value in updates.items():
            self._banks[i]['w'] = value
        for i in receipts:
            self._receipts[i] = dict(self._receipts[i], consumed=True)
        if self.observer is not None:
            self.observer('waive' if deltas is None else 'feedback',
                          dict(unit=self, receipts_input=receipts, deltas=deltas))

    def rollback(self):
        if self._active is None:
            raise RuntimeError('no active transaction')
        self._active = None
        self._emit('rollback')

    def detach_state(self):
        if self._active is not None:
            raise RuntimeError('detach only between transactions')
        self._banks = {i: {k: v.detach() for k, v in bank.items()} for i, bank in self._banks.items()}
        self._training_transactions = 0
        self._emit('detach')

    def clear_state(self):
        if self._active is not None:
            raise RuntimeError('rollback or commit before clear')
        self._banks, self._receipts = {}, {}
        self._training_transactions = 0
        self._emit('clear')

    def export_state(self):
        """Detached checkpoint of committed state; active transactions cannot be saved."""
        if self._active is not None:
            raise RuntimeError('finish transaction before checkpoint')
        return dict(version=1, banks={i: {k: v.detach().clone() for k, v in bank.items()}
                                     for i, bank in self._banks.items()}, receipts=deepcopy(self._receipts))

    def restore_state(self, state):
        if self._active is not None or self._training_transactions:
            raise RuntimeError('restore requires idle detached state')
        if not isinstance(state, dict) or set(state) != {'version', 'banks', 'receipts'} or state['version'] != 1:
            raise ValueError('invalid state envelope')
        banks, receipts = state['banks'], state['receipts']
        if not isinstance(banks, dict) or not isinstance(receipts, dict) or len(banks) > 4 or set(banks) != set(receipts):
            raise ValueError('invalid state contexts')
        copied = {}
        for i, bank in banks.items():
            if not isinstance(i, str) or not i or not isinstance(bank, dict) or set(bank) != {'c', 'w', 'e'}:
                raise ValueError('invalid bank')
            copied[i] = {}
            for k, v in bank.items():
                if not isinstance(v, torch.Tensor) or v.shape != (self.out_features,) or v.dtype != self.weight.dtype or v.device != self.weight.device or not torch.isfinite(v).all():
                    raise ValueError('invalid bank tensor')
                if (v.abs() > 1).any():
                    raise ValueError('state exceeds bounded range')
                copied[i][k] = v.detach().clone()
            r = receipts[i]
            if not isinstance(r, dict) or set(r) != {'event_id', 'consumed'} or not isinstance(r['event_id'], str) or not r['event_id'] or type(r['consumed']) is not bool:
                raise ValueError('invalid receipt')
        self._banks, self._receipts = copied, deepcopy(receipts)
        self._emit('restore')

    def get_extra_state(self):
        return self.export_state()

    def set_extra_state(self, state):
        self.restore_state(state)
