"""Stage53 migration; frozen stage51 graph is intentionally NOT attached.

Stage51 clear_cache cannot release retained observation graphs safely. Combined
cross-layer graph and two-event training requires a later graph version.
"""
import math
import torch
from torch import nn
from neural_dynamics.fused_qwen import FusedQwenController
from neural_dynamics.dendritic_cell import DendriticLinear, SLOW_NAMES


class DendriticQwenController(FusedQwenController):
    def __init__(self, units):
        self.units = dict(units)
        if not self.units or len({id(u) for u in self.units.values()}) != len(self.units):
            raise ValueError('unique nonempty units required')
        if any(type(u) is not DendriticLinear for u in self.units.values()):
            raise TypeError('all units must be stage53 DendriticLinear')
        self.active = None
        self.counts = {k: 0 for k in ('begin', 'commit', 'feedback', 'waive', 'rollback', 'detach', 'clear', 'restore')}

    def set_gains(self, output_gain):
        if self.active is not None or any(u._training_transactions for u in self.units.values()):
            raise RuntimeError('gain change requires idle detached state')
        if isinstance(output_gain, bool) or not isinstance(output_gain, (int, float)) or not math.isfinite(output_gain):
            raise ValueError('finite prescribed gain required')
        with torch.no_grad():
            for u in self.units.values():
                u.output_gain.fill_(output_gain)

    def slow_state(self):
        return {n: {k: getattr(u, k).detach().clone() for k in SLOW_NAMES} for n, u in self.units.items()}


def migrate_projections(model, inventory, *, branches=4, observer_factory=None):
    targets = inventory['targets']
    names = [t['name'] for t in targets]
    if not names or len(names) != len(set(names)) or type(branches) is not int or branches < 1:
        raise ValueError('unique targets and positive branch count required')
    seen = set()
    all_names = list(model.named_parameters(remove_duplicate=False))
    for t in targets:
        old = model.get_submodule(t['name'])
        if type(old) is not nn.Linear or list(old.weight.shape) != t['weight_shape'] or old.in_features < branches or old.weight.device.type != 'cpu' or old.weight.dtype != torch.float32:
            raise ValueError('migration target type/shape/device/dtype mismatch: ' + t['name'])
        for p in old.parameters():
            if id(p) in seen or sum(candidate is p for _, candidate in all_names) != 1:
                raise ValueError('projection ties require separate design')
            seen.add(id(p))
    del all_names, old, p
    units = {}
    for name in names:
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child)
        unit = DendriticLinear(old, branches=branches, observer=None if observer_factory is None else observer_factory(name))
        unit.train(old.training)
        setattr(parent, child, unit)
        units[name] = unit
        del old
    model.requires_grad_(True)
    return DendriticQwenController(units)
