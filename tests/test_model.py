"""Unit tests for PPGNet architecture and gradient optimization."""

import torch
import torch.nn as nn
import torch.optim as optim
import pytest

from model import PPGNet


def test_ppgnet_output_shape():
    """Verify forward pass output dimensions for batch of PPG maps."""
    model = PPGNet()
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 128)
    logits = model(x)
    assert logits.shape == (batch_size,)


def test_ppgnet_loss_descent():
    """Verify that 20 Adam steps on a single fixed batch reduce BCE loss."""
    torch.manual_seed(42)
    model = PPGNet()
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(8, 3, 32, 128)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    initial_loss = None
    final_loss = None

    for step in range(20):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == 19:
            final_loss = loss.item()

    assert final_loss < initial_loss
    assert final_loss < 0.1  # Significant convergence on single batch
