"""Whole-network projection migration and graph-safe atomic state controller.

This module imports only Torch and the frozen stage46 unit. The caller owns the
model forward and disables KV caching; every unit shares a transaction mask.
Atomic recovery copies containers while retaining Tensor identity/autograd
references. It never deep-copies nonleaf tensors or a model.
"""
from __future__ import annotations
from collections.abc import Mapping
import hashlib
import torch
from torch import nn
from neural_dynamics.fused_unit import FusedLinear


def reference_copy(value):
    """Copy lifecycle containers, preserving every tensor/graph reference."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        return {k: reference_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [reference_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(reference_copy(v) for v in value)
    return value


def tensor_fingerprint(tensor):
    value = tensor.detach().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy()).hexdigest())


def parameter_fingerprints(model, names=None):
    wanted = None if names is None else set(names)
    result = {n: tensor_fingerprint(p) for n, p in model.named_parameters(remove_duplicate=False)
              if wanted is None or n in wanted}
    if wanted is not None and set(result) != wanted:
        raise AssertionError('inherited named parameter coverage changed')
    return result


def tied_parameter_groups(model):
    groups = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(param), []).append(name)
    return sorted(sorted(names) for names in groups.values() if len(names) > 1)


def migrate_projections(model, inventory, *, observer_factory=None):
    """Validate first; replace/release each Linear individually, preserving names.

    Projection tying is rejected before mutation; replacing tied projections
    would need a separate migration design. Nonprojection ties remain untouched.
    """
    targets = inventory['targets']
    names = [t['name'] for t in targets]
    if not names or len(names) != len(set(names)):
        raise ValueError('nonempty unique inventory required')
    seen = set()
    for target in targets:
        old = model.get_submodule(target['name'])
        if not isinstance(old, nn.Linear) or list(old.weight.shape) != target['weight_shape']:
            raise ValueError('target type/shape mismatch: ' + target['name'])
        if old.weight.device.type != 'cpu' or old.weight.dtype != torch.float32:
            raise ValueError('migration requires CPU float32')
        for p in old.parameters():
            if id(p) in seen:
                raise ValueError('tied projections require a separate design')
            seen.add(id(p))
    del old, p
    units = {}
    for name in names:
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child)
        unit = FusedLinear(old, observer=None if observer_factory is None else observer_factory(name))
        unit.train(old.training)
        setattr(parent, child, unit)
        units[name] = unit
        del old  # no retained original model or original projection bank
    model.requires_grad_(True)
    return FusedQwenController(units)


class FusedQwenController:
    def __init__(self, units: Mapping[str, FusedLinear]):
        self.units = dict(units)
        if not self.units or len({id(u) for u in self.units.values()}) != len(self.units):
            raise ValueError('unique nonempty unit map required')
        if any(not isinstance(u, FusedLinear) for u in self.units.values()):
            raise TypeError('all units must be frozen stage46 FusedLinear')
        self.active = None
        self.counts = {k: 0 for k in ('begin', 'commit', 'feedback', 'waive', 'rollback', 'detach', 'clear', 'restore')}

    def _checkpoint_references(self):
        return {n: dict(banks=reference_copy(u._banks), receipts=reference_copy(u._receipts),
                        active=reference_copy(u._active), training=u._training_transactions)
                for n, u in self.units.items()}

    def _recover(self, saved):
        for name, state in saved.items():
            u = self.units[name]
            u._banks, u._receipts, u._active = state['banks'], state['receipts'], state['active']
            u._training_transactions = state['training']

    def _atomic(self, operation, action):
        saved = self._checkpoint_references()
        active = reference_copy(self.active)
        try:
            result = action()
        except BaseException:
            self._recover(saved)
            self.active = active
            raise
        self.counts[operation] += 1
        return result

    def begin(self, context_ids, event_id, last_valid_mask, *, training=False):
        if self.active is not None:
            raise RuntimeError('global transaction already active')
        ids = tuple(context_ids)
        def action():
            for u in self.units.values():
                u.begin(ids, event_id, last_valid_mask, training=training)
            self.active = dict(context_ids=ids, event_id=event_id, training=training)
        self._atomic('begin', action)

    def commit(self):
        if self.active is None:
            raise RuntimeError('no global active transaction')
        def action():
            receipts = {name: u.commit() for name, u in self.units.items()}
            self.active = None
            return receipts
        return self._atomic('commit', action)

    def _feedback(self, receipts, deltas):
        if self.active is not None or not isinstance(receipts, dict) or set(receipts) != set(self.units):
            raise ValueError('idle global controller and complete module receipts required')
        def action():
            for name, u in self.units.items():
                if deltas is None:
                    u.waive_feedback(receipts[name])
                else:
                    u.apply_feedback(receipts[name], deltas)
        self._atomic('waive' if deltas is None else 'feedback', action)

    def apply_feedback(self, receipts, deltas):
        self._feedback(receipts, deltas)

    def waive_feedback(self, receipts):
        self._feedback(receipts, None)

    def rollback(self):
        if self.active is None:
            raise RuntimeError('no global active transaction')
        def action():
            for u in self.units.values():
                u.rollback()
            self.active = None
        self._atomic('rollback', action)

    def detach_state(self):
        def action():
            for u in self.units.values():
                u.detach_state()
        self._atomic('detach', action)

    def clear_state(self):
        def action():
            for u in self.units.values():
                u.clear_state()
        self._atomic('clear', action)

    def export_state(self):
        if self.active is not None:
            raise RuntimeError('finish global transaction before save')
        return {name: u.export_state() for name, u in self.units.items()}

    def restore_state(self, state):
        if not isinstance(state, dict) or set(state) != set(self.units):
            raise ValueError('complete module state required')
        def action():
            for name, u in self.units.items():
                u.restore_state(state[name])
        self._atomic('restore', action)

    def set_gains(self, alpha, beta):
        if self.active is not None:
            raise RuntimeError('gains may change only between global transactions')
        if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)) or not torch.isfinite(torch.tensor([alpha, beta])).all():
            raise ValueError('finite prescribed gains required')
        with torch.no_grad():
            for u in self.units.values():
                u.alpha.fill_(alpha)
                u.beta.fill_(beta)

    def slow_state(self):
        return {n: {k: getattr(u, k).detach().clone() for k in ('alpha', 'beta', 'retention_logits')}
                for n, u in self.units.items()}
