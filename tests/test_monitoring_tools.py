import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytest

from hex_ai.training_utils import GradientMonitor, ActivationMonitor

class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.value_head = nn.Linear(4, 1)
        self.policy_head = nn.Linear(4, 2)
    def forward(self, x):
        h = torch.relu(self.shared(x))
        return self.policy_head(h), self.value_head(h)

def test_gradient_monitor_logs_nonzero_gradients():
    model = DummyNet()
    monitor = GradientMonitor(model, log_interval=1)
    x = torch.randn(8, 4)
    policy_pred, value_pred = model(x)
    policy_target = torch.zeros(8, 2)
    value_target = torch.ones(8, 1)
    loss = (policy_pred**2).mean() + (value_pred**2).mean()
    loss.backward()
    monitor.log_gradients(batch_idx=0)
    norms = monitor.compute_gradient_norms()
    # At least one head should have nonzero gradients
    assert any(abs(v) > 0 for v in norms.values())

def test_activation_monitor_logs_activations():
    model = DummyNet()
    monitor = ActivationMonitor(model, log_interval=1)
    x = torch.randn(8, 4)
    policy_pred, value_pred = model(x)
    monitor.log_activations(batch_idx=0)
    # Should have logged activations for value_head and policy_head
    found = False
    for k in monitor.activation_history:
        if 'value_head' in k or 'policy_head' in k:
            found = True
            assert len(monitor.activation_history[k]) > 0
    assert found

def test_monitoring_callback_collects_varying_values():
    model = DummyNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    grad_monitor = GradientMonitor(model, log_interval=1)
    act_monitor = ActivationMonitor(model, log_interval=1)
    grad_history = []
    act_history = []
    for step in range(5):
        x = torch.randn(8, 4)
        policy_pred, value_pred = model(x)
        loss = (policy_pred**2).mean() + (value_pred**2).mean()
        optimizer.zero_grad()
        loss.backward()
        grad_monitor.log_gradients(batch_idx=step)
        act_monitor.log_activations(batch_idx=step)
        grad_norms = grad_monitor.compute_gradient_norms()
        grad_history.append(grad_norms['value_head'])
        act_stats = {}
        for layer_name, acts in act_monitor.activation_history.items():
            if 'value_head' in layer_name and acts:
                act_stats[layer_name] = acts[-1]['mean']
        act_history.append(act_stats)
        optimizer.step()
    # Check that not all gradient norms are the same
    assert len(set(grad_history)) > 1
    # Check that at least some activation means vary
    all_means = [list(d.values())[0] for d in act_history if d]
    assert len(set(all_means)) > 1 