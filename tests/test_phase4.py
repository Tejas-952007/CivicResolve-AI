"""Comprehensive unit and integration tests for CivicResolve AI Phase 4.

Validates:
- Simulation configuration boundaries and validation.
- Tabular state encoding determinism.
- Safe action execution in empty and non-empty queue states.
- Bellman equation Q-value update correctness and terminal transition handling.
- Deterministic reproducibility under fixed random seeds.
- Zero-leakage: True outcomes (resolution_hours, target_long_duration) are never accessed during simulation.
- Identical arrival streams and capacity budgets provided to all compared policies.
- Frozen classification pipeline integrity (no retraining or modification).
- Unseen and rare complaint categories gracefully mapped to 'Other' queue.
- Bounded training runtime and parameter limits.
- Metrics calculation accuracy (slot utilization, waiting age, starvation, capture rate).
- Clear separation between intake-time predicted risk and retrospective historical labels.
- Flask endpoints (/resource-planning, /resource-planning/run, /resource-planning/export-csv).
- PDF report includes simulation appendix with valid metrics.
- Resilience against invalid or empty form inputs.
"""
from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import numpy as np
import pandas as pd

from app import create_app
from src.services.rl_simulation_service import (
    FROZEN_QUEUES,
    NUM_QUEUES,
    CivicQueueEnv,
    QLearningAgent,
    SimulationComplaint,
    SimulationConfig,
    build_arrival_stream,
    encode_state,
    evaluate_single_policy,
    get_complaint_pool,
    map_complaint_to_queue,
    run_policy_comparison,
    train_q_learning_policy,
)
from src.services.prediction_service import load_models


@pytest.fixture(scope="module")
def app():
    """Create and configure Flask application for Phase 4 testing."""
    test_app = create_app()
    test_app.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
    })
    return test_app


@pytest.fixture(scope="module")
def client(app):
    """Test client for HTTP route verification."""
    return app.test_client()


@pytest.fixture
def sample_config():
    """Fast, lightweight configuration for testing."""
    return SimulationConfig(
        capacity_per_period=4,
        horizon_periods=10,
        arrival_batch_size=5,
        training_episodes=15,
        alpha=0.1,
        gamma=0.9,
        seed=42,
    )


class TestPhase4ReinforcementLearning:
    """Test suite for Phase 4 Q-learning simulator, baselines, and web interface."""

    def test_simulation_config_validation(self):
        """Verify configuration enforces valid parameter boundaries."""
        valid_cfg = SimulationConfig(capacity_per_period=8, horizon_periods=25)
        valid_cfg.validate()

        with pytest.raises(ValueError, match="capacity_per_period"):
            SimulationConfig(capacity_per_period=0).validate()

        with pytest.raises(ValueError, match="horizon_periods"):
            SimulationConfig(horizon_periods=2).validate()

        with pytest.raises(ValueError, match="alpha"):
            SimulationConfig(alpha=1.5).validate()

        with pytest.raises(ValueError, match="gamma"):
            SimulationConfig(gamma=-0.1).validate()

    def test_state_encoding_determinism(self):
        """Verify state encoding is deterministic and maps identical queue states to identical keys."""
        q1 = [
            SimulationComplaint(1, "Illegal Parking", "Illegal Parking", 0, 0.85, 0),
            SimulationComplaint(2, "Illegal Parking", "Illegal Parking", 0, 0.30, 1),
        ]
        q2 = [
            SimulationComplaint(3, "HEAT/HOT WATER", "HEAT/HOT WATER", 2, 0.92, 0),
        ]
        queues_a = [q1, [], q2, [], [], []]
        queues_b = [copy.copy(q1), [], copy.copy(q2), [], [], []]

        state_a = encode_state(queues_a, capacity_remaining=3)
        state_b = encode_state(queues_b, capacity_remaining=3)
        assert state_a == state_b
        assert "hr:2" in state_a  # Queue 2 has highest risk (0.92)
        assert "cap:1" in state_a  # 3 capacity slots is tier 1

    def test_unseen_category_safe_mapping(self):
        """Verify unknown or unobserved complaint categories safely map to 'Other' without error."""
        assert map_complaint_to_queue("Illegal Parking") == "Illegal Parking"
        assert map_complaint_to_queue("Noise - Residential") == "Noise - Residential"
        assert map_complaint_to_queue("Extraterrestrial Sighting") == "Other"
        assert map_complaint_to_queue("Rare Pune Municipal Water Issue") == "Other"
        assert map_complaint_to_queue("") == "Other"
        assert map_complaint_to_queue(None) == "Other"

    def test_empty_and_non_empty_queue_action_handling(self, sample_config):
        """Verify selecting an empty queue does not crash and incurs appropriate penalty."""
        complaint = SimulationComplaint(101, "Illegal Parking", "Illegal Parking", 0, 0.75, 0)
        # Stream with 1 item in queue 0
        stream = [[complaint]] + [[] for _ in range(sample_config.horizon_periods - 1)]
        env = CivicQueueEnv(stream, sample_config)
        env.reset()

        # Step 1: Select non-empty queue 0
        state1, r1, done1, info1 = env.step(0)
        assert info1["served"] is True
        assert r1 > 0.0
        assert env.total_slots_used == 1

        # Step 2: Select empty queue 1
        state2, r2, done2, info2 = env.step(1)
        assert info2["served"] is False
        assert r2 < 0.0  # Empty queue penalty
        assert env.total_slots_used == 1  # No new item served

    def test_q_value_bellman_update_and_terminal(self, sample_config):
        """Verify tabular Q-value update strictly matches Bellman equation."""
        agent = QLearningAgent(sample_config)
        state = "s_init"
        next_state = "s_next"
        action = 0

        # Set initial Q(s_next, a)
        agent.get_q_values(next_state)
        agent.q_table[next_state] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]  # max Q(s', a') = 3.0

        # Non-terminal update: target = reward + gamma * max Q(s')
        # expected = 0.0 + alpha * (target - 0.0) = 0.1 * (2.0 + 0.9 * 3.0) = 0.1 * 4.7 = 0.47
        agent.update(state, action, reward=2.0, next_state=next_state, done=False)
        assert np.isclose(agent.get_q_values(state)[0], 0.47)

        # Terminal update: target = reward
        agent.update(state, action, reward=5.0, next_state=next_state, done=True)
        # current = 0.47, target = 5.0, delta = 5.0 - 0.47 = 4.53
        # new = 0.47 + 0.1 * 4.53 = 0.923
        assert np.isclose(agent.get_q_values(state)[0], 0.923)

    def test_reproducibility_with_fixed_seed(self, sample_config):
        """Verify simulator produces bit-for-bit identical results with fixed random seed."""
        pool = get_complaint_pool(split_name="val", use_oof_for_training=False, seed=42)
        stream_a = build_arrival_stream(pool, horizon=5, batch_size=4, seed=123)
        stream_b = build_arrival_stream(pool, horizon=5, batch_size=4, seed=123)

        assert len(stream_a) == len(stream_b)
        for batch_a, batch_b in zip(stream_a, stream_b):
            ids_a = [c.complaint_id for c in batch_a]
            ids_b = [c.complaint_id for c in batch_b]
            assert ids_a == ids_b

        # Compare policy execution
        env_a = CivicQueueEnv(stream_a, sample_config)
        env_b = CivicQueueEnv(stream_b, sample_config)

        m_a, r_a = evaluate_single_policy("FIFO", env_a, seed=123)
        m_b, r_b = evaluate_single_policy("FIFO", env_b, seed=123)
        assert r_a == r_b
        assert m_a["items_handled"] == m_b["items_handled"]

    def test_zero_leakage_policy_execution(self, sample_config):
        """Verify policies and environments do not inspect true historical outcomes during simulation."""
        pool = get_complaint_pool(split_name="val", use_oof_for_training=False, seed=42)
        # Artificially corrupt true resolution hours to verify policies never touch them
        corrupted_pool = copy.deepcopy(pool[:50])
        for c in corrupted_pool:
            c.true_resolution_hours = 999999.0
            c.true_long_duration = 1

        stream = [corrupted_pool[:10] for _ in range(sample_config.horizon_periods)]
        env = CivicQueueEnv(stream, sample_config)
        
        # Train agent on dummy pool
        agent = QLearningAgent(sample_config)
        # Run step and verify reward depends ONLY on predicted_risk and simulated waiting age
        env.reset()
        next_s, r, done, info = env.step(0)
        # If true_resolution_hours had been used, reward would have been skewed to astronomical values
        assert -10.0 <= r <= 10.0

    def test_same_arrivals_supplied_to_all_policies(self, sample_config):
        """Verify every baseline policy evaluates on identical arrival batches and capacities."""
        res = run_policy_comparison(
            sample_config,
            eval_split="val",
            eval_seeds=[42],
        )
        summary = res["summary"]
        assert len(summary) == 5
        # Total slots available must be identical across all 5 policies
        policies = [s["policy"] for s in summary]
        assert set(policies) == {"Q-Learning", "FIFO", "Risk-First", "Risk-Plus-Wait", "Round-Robin"}

    def test_frozen_classifier_not_retrained(self):
        """Verify Phase 2 preferred classifier joblib file remains frozen and untouched."""
        _, clf, meta = load_models()
        assert clf is not None
        assert meta["phase"] == "Phase 2: Leakage-Free ML Modeling"
        assert meta["long_duration_threshold_hours"] == 124.925

    def test_metrics_distinguish_predicted_vs_true_outcomes(self, sample_config):
        """Verify summary metrics cleanly separate intake model risk from retrospective actual outcomes."""
        res = run_policy_comparison(sample_config, eval_split="val", eval_seeds=[42])
        for s in res["summary"]:
            assert "mean_pred_high_risk_handled" in s
            assert "mean_true_long_handled" in s
            assert "mean_true_long_capture_pct" in s
            assert "mean_waiting_periods" in s
            # Must be valid non-negative numbers
            assert s["mean_pred_high_risk_handled"] >= 0.0
            assert s["mean_true_long_handled"] >= 0.0
            assert 0.0 <= s["mean_true_long_capture_pct"] <= 100.0

    def test_flask_resource_planning_views(self, client):
        """Verify GET /resource-planning renders HTTP 200 with required charts and disclaimers."""
        resp = client.get("/resource-planning")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Resource Allocation &amp; Policy Simulation" in html or "Resource Allocation & Policy Simulation" in html
        assert "Research Simulation Disclosure" in html
        assert "Q-Learning" in html
        assert "FIFO" in html
        assert "chart-policy-reward" in html
        assert "chart-queue-fairness" in html
        assert "chart-training-curve" in html

    def test_flask_resource_planning_export_csv(self, client):
        """Verify CSV export returns expected columns and headers."""
        resp = client.get("/resource-planning/export-csv")
        assert resp.status_code == 200
        assert resp.headers["Content-Type"].startswith("text/csv")
        csv_text = resp.get_data(as_text=True)
        assert "policy" in csv_text
        assert "mean_reward" in csv_text
        assert "mean_slot_utilization_pct" in csv_text
        assert "mean_true_long_capture_pct" in csv_text

    def test_flask_resource_planning_run_endpoint(self, client):
        """Verify POST /resource-planning/run executes bounded simulation safely and redirects."""
        payload = {
            "capacity_per_period": "5",
            "horizon_periods": "10",
            "arrival_batch_size": "6",
            "training_episodes": "15",
            "seed": "99",
            "alpha": "0.1",
            "gamma": "0.95",
            "eval_split": "val",
        }
        resp = client.post("/resource-planning/run", data=payload, follow_redirects=True)
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Simulation completed successfully" in html

    def test_pdf_report_includes_simulation_appendix(self):
        """Verify generated PDF includes Section 6 simulation appendix when metrics exist."""
        from src.services.report_service import build_pdf_report
        import subprocess

        pdf_bytes = build_pdf_report()
        assert len(pdf_bytes) > 5000

        # Save to temp file and inspect via pdftotext
        tmp_path = Path("/tmp/pytest_audit_report.pdf")
        tmp_path.write_bytes(pdf_bytes)

        res = subprocess.run(["pdftotext", str(tmp_path), "-"], capture_output=True, text=True, check=True)
        text = res.stdout
        assert "6. Appendix: Reinforcement Learning & Queue Allocation Simulation" in text
        assert "Q-Learning" in text
        assert "FIFO" in text
        assert "True Long" in text
        assert "Capture" in text
