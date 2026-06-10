"""
test_project.py
===============

pytest suite for the Cantor Approximation Lower Bound proof project.

Run with:
    source ~/heartlib/.venv/bin/activate
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "empirical"))
from verify import (
    generate_level_n_intervals,
    cantor_level_n,
    construct_exact_relu,
    construct_limited_relu,
    relu_cantor_error,
    relu_exact_error,
    count_pieces,
    check_theorem_1,
    check_theorem_2,
    check_theorem_3,
    manual_seed,
    get_device,
)


@pytest.fixture(scope="module", autouse=True)
def seed():
    manual_seed(1729)


class TestCantorFunction:
    """Unit tests for the level-n Cantor approximation."""

    def test_n1_has_3_intervals(self):
        intervals = generate_level_n_intervals(1)
        assert len(intervals) == 3

    def test_n2_has_9_intervals(self):
        intervals = generate_level_n_intervals(2)
        assert len(intervals) == 9

    def test_n3_has_27_intervals(self):
        intervals = generate_level_n_intervals(3)
        assert len(intervals) == 27

    def test_boundary_values(self):
        x = np.array([0.0, 1.0])
        y = cantor_level_n(3, x)
        assert y[0] == 0.0
        assert y[1] == 1.0

    def test_monotonic(self):
        x = np.linspace(0, 1, 100)
        y = cantor_level_n(2, x)
        assert np.all(np.diff(y) >= -1e-10)


class TestReLUConstruction:
    """Tests for ReLU exact match construction."""

    def test_exact_match_n1(self):
        n = 1
        x = np.linspace(0, 1, 100)
        y_target = cantor_level_n(n, x)
        
        net = construct_exact_relu(n)
        device = get_device()
        x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(device)
        
        with torch.no_grad():
            y_relu = net(x_t).cpu().numpy().flatten()
        
        error = np.abs(y_target - y_relu).max()
        assert error < 1e-4

    def test_exact_match_n2(self):
        n = 2
        x = np.linspace(0, 1, 100)
        y_target = cantor_level_n(n, x)
        
        net = construct_exact_relu(n)
        device = get_device()
        x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(device)
        
        with torch.no_grad():
            y_relu = net(x_t).cpu().numpy().flatten()
        
        error = np.abs(y_target - y_relu).max()
        assert error < 1e-4

    def test_under_capacity_error(self):
        """Under-capacity network has nonzero error."""
        n = 2
        exact_H = 3 ** n
        under_H = exact_H - 1
        
        exact_err = relu_exact_error(n)
        under_err = relu_cantor_error(under_H, n)
        
        assert exact_err < 1e-3
        assert under_err > 1e-3

    def test_piece_count_match(self):
        """Exact ReLU has same number of pieces as c_n."""
        n = 2
        x = np.linspace(0, 1, 5000)
        
        y_cantor = cantor_level_n(n, x)
        n_cantor = count_pieces(y_cantor)
        
        net = construct_exact_relu(n)
        device = get_device()
        x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(device)
        with torch.no_grad():
            y_relu = net(x_t).cpu().numpy().flatten()
        n_relu = count_pieces(y_relu)
        
        assert n_cantor == n_relu


class TestTheorem1:
    def test_pass(self):
        r = check_theorem_1()
        assert r.passed, f"Theorem 1 failed: {r.detail}"


class TestTheorem2:
    def test_pass(self):
        r = check_theorem_2()
        assert r.passed, f"Theorem 2 failed: {r.detail}"


class TestTheorem3:
    def test_pass(self):
        r = check_theorem_3()
        assert r.passed, f"Theorem 3 failed: {r.detail}"
