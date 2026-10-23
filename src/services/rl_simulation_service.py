"""CivicResolve AI — Phase 4: Reinforcement Learning & Queue Allocation Service.

Simulates limited municipal complaint-review / dispatch capacity across discrete complaint queues.
Implements:
- Tabular Q-learning with explicit state encoding and transparent reward design.
- Benchmark policies: FIFO, Risk-First, Risk-Plus-Wait, and Round-Robin.
- Strict data contract: Intake-time features only; zero visibility into post-resolution outcomes.
- Retrospective (post-simulation) comparison of true long-duration capture without counterfactual claims.
- Multi-seed repeatability and statistical variance tracking.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.features import (
    INTAKE_FEATURE_ALLOWLIST,
    compute_training_threshold,
    load_dataset,
    prepare_feature_matrix,
    split_temporal_data,
)
from src.preprocess import build_preprocessor
from src.services.prediction_service import (
    LONG_DURATION_THRESHOLD_HOURS,
    load_models,
)

logger = logging.getLogger("rl_simulation_service")

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_PATH = BASE_DIR / "data" / "processed" / "nyc_311_phase1_profiled.csv"

# Top 5 categories strictly established from 2023 training data, remainder mapped to 'Other'
FROZEN_QUEUES = [
    "Illegal Parking",
    "Noise - Residential",
    "HEAT/HOT WATER",
    "Blocked Driveway",
    "Noise - Street/Sidewalk",
    "Other",
]
NUM_QUEUES = len(FROZEN_QUEUES)
QUEUE_TO_IDX = {q: idx for idx, q in enumerate(FROZEN_QUEUES)}
IDX_TO_QUEUE = {idx: q for idx, q in enumerate(FROZEN_QUEUES)}


def map_complaint_to_queue(complaint_type: str) -> str:
    """Map any complaint category safely to the 6 discrete queues."""
    if not isinstance(complaint_type, str):
        return "Other"
    clean = complaint_type.strip()
    return clean if clean in QUEUE_TO_IDX else "Other"


@dataclass
class SimulationComplaint:
    """Single complaint item entering the simulated queue."""
    complaint_id: int
    complaint_type: str
    queue_name: str
    queue_idx: int
    predicted_risk: float
    arrival_period: int
    borough: str = "Unspecified"
    true_resolution_hours: float | None = None
    true_long_duration: int | None = None


@dataclass
class SimulationConfig:
    """Configuration hyperparameters for the municipal queue allocation simulator."""
    capacity_per_period: int = 8
    horizon_periods: int = 25
    arrival_batch_size: int = 10
    training_episodes: int = 150
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.98
    seed: int = 42
    reward_weight_risk: float = 2.0
    reward_weight_wait: float = 1.0
    reward_weight_backlog: float = 0.15
    reward_penalty_empty: float = 0.5
    reward_penalty_starve: float = 0.4
    reward_penalty_term: float = 0.5
    use_oof_for_training: bool = True

    def validate(self) -> None:
        """Validate parameter boundaries for safe execution."""
        if not (1 <= self.capacity_per_period <= 50):
            raise ValueError(f"capacity_per_period ({self.capacity_per_period}) must be between 1 and 50.")
        if not (5 <= self.horizon_periods <= 100):
            raise ValueError(f"horizon_periods ({self.horizon_periods}) must be between 5 and 100.")
        if not (1 <= self.arrival_batch_size <= 50):
            raise ValueError(f"arrival_batch_size ({self.arrival_batch_size}) must be between 1 and 50.")
        if not (1 <= self.training_episodes <= 500):
            raise ValueError(f"training_episodes ({self.training_episodes}) must be between 1 and 500.")
        if not (0.01 <= self.alpha <= 1.0):
            raise ValueError(f"alpha ({self.alpha}) must be between 0.01 and 1.0.")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma ({self.gamma}) must be between 0.0 and 1.0.")
        if self.seed < 0:
            raise ValueError(f"seed ({self.seed}) must be non-negative.")
        for rw in [
            "reward_weight_risk",
            "reward_weight_wait",
            "reward_weight_backlog",
            "reward_penalty_empty",
            "reward_penalty_starve",
            "reward_penalty_term",
        ]:
            val = getattr(self, rw)
            if not np.isfinite(val) or val < 0.0:
                raise ValueError(f"{rw} ({val}) must be a non-negative finite number.")


# ---------------- State Encoding ----------------

def encode_state(
    queues: list[list[SimulationComplaint]],
    capacity_remaining: int,
) -> str:
    """Encode continuous queue status into a compact, discrete tabular state string.
    
    Components:
    1. highest_risk_queue (hr): queue index containing the complaint with highest predicted risk (-1..5)
    2. oldest_queue (old): queue index containing the complaint waiting the longest (-1..5)
    3. capacity_tier (cap): 0 (<= 1 slot), 1 (2-3 slots), 2 (4+ slots)
    4. backlog_tier (bl): 0 (0-5 items), 1 (6-15 items), 2 (16-30 items), 3 (31+ items)
    5. risk_pressure (rp): 0 (< 25% high-risk), 1 (25-50% high-risk), 2 (> 50% high-risk)
    """
    total_items = sum(len(q) for q in queues)
    
    if total_items == 0:
        hr_q = -1
        old_q = -1
        rp = 0
    else:
        best_risk = -1.0
        hr_q = -1
        oldest_period = 999999
        old_q = -1
        high_risk_count = 0
        
        for q_idx, q in enumerate(queues):
            if not q:
                continue
            for item in q:
                if item.predicted_risk > best_risk:
                    best_risk = item.predicted_risk
                    hr_q = q_idx
                if item.arrival_period < oldest_period:
                    oldest_period = item.arrival_period
                    old_q = q_idx
                if item.predicted_risk >= 0.5:
                    high_risk_count += 1
                    
        ratio = high_risk_count / total_items
        if ratio < 0.25:
            rp = 0
        elif ratio <= 0.50:
            rp = 1
        else:
            rp = 2

    # Capacity bin
    if capacity_remaining <= 1:
        cap_t = 0
    elif capacity_remaining <= 3:
        cap_t = 1
    else:
        cap_t = 2

    # Backlog bin
    if total_items <= 5:
        bl_t = 0
    elif total_items <= 15:
        bl_t = 1
    elif total_items <= 30:
        bl_t = 2
    else:
        bl_t = 3

    return f"hr:{hr_q}_old:{old_q}_cap:{cap_t}_bl:{bl_t}_rp:{rp}"


# ---------------- Environment ----------------

class CivicQueueEnv:
    """Discrete municipal complaint queue resource allocation environment."""

    def __init__(
        self,
        arrival_stream: list[list[SimulationComplaint]],
        config: SimulationConfig,
    ) -> None:
        self.arrival_stream = arrival_stream
        self.config = config
        self.horizon = min(len(arrival_stream), config.horizon_periods)
        self.current_period = 0
        self.capacity_remaining = 0
        self.queues: list[list[SimulationComplaint]] = [[] for _ in range(NUM_QUEUES)]
        self.total_slots_used = 0
        self.total_slots_available = self.horizon * config.capacity_per_period
        
        # Logging & tracking
        self.handled_items: list[dict[str, Any]] = []
        self.period_metrics: list[dict[str, Any]] = []
        self.starvation_tracker: list[int] = [0 for _ in range(NUM_QUEUES)]
        self.max_starvation_observed: list[int] = [0 for _ in range(NUM_QUEUES)]

    def reset(self) -> str:
        """Reset environment to the beginning of the horizon."""
        self.current_period = 0
        self.queues = [[] for _ in range(NUM_QUEUES)]
        self.handled_items = []
        self.period_metrics = []
        self.starvation_tracker = [0 for _ in range(NUM_QUEUES)]
        self.max_starvation_observed = [0 for _ in range(NUM_QUEUES)]
        self.total_slots_used = 0
        
        # Ingest period 0 arrival batch
        self._ingest_current_period_arrivals()
        self.capacity_remaining = self.config.capacity_per_period
        return encode_state(self.queues, self.capacity_remaining)

    def _ingest_current_period_arrivals(self) -> None:
        """Add incoming complaints of the current period to queues."""
        if self.current_period < len(self.arrival_stream):
            batch = self.arrival_stream[self.current_period]
            for item in batch:
                # Stamp arrival period
                item_copy = copy.copy(item)
                item_copy.arrival_period = self.current_period
                self.queues[item_copy.queue_idx].append(item_copy)

    def step(self, action: int) -> tuple[str, float, bool, dict[str, Any]]:
        """Execute one allocation action.
        
        Action 0..5: Select complaint from corresponding queue (FIFO within queue).
        Returns: (next_state, reward, done, info)
        """
        assert 0 <= action < NUM_QUEUES, f"Invalid action {action}"

        served_item: SimulationComplaint | None = None
        reward = 0.0

        if len(self.queues[action]) > 0:
            # FIFO selection within the designated queue
            served_item = self.queues[action].pop(0)
            waiting_age = self.current_period - served_item.arrival_period
            
            # Reset starvation for this queue
            self.starvation_tracker[action] = 0
            
            # Reward: high predicted risk + waiting age
            norm_wait = min(waiting_age, 5) / 5.0
            reward += (self.config.reward_weight_risk * served_item.predicted_risk)
            reward += (self.config.reward_weight_wait * norm_wait)
            
            self.handled_items.append({
                "complaint_id": served_item.complaint_id,
                "complaint_type": served_item.complaint_type,
                "queue_name": served_item.queue_name,
                "queue_idx": served_item.queue_idx,
                "predicted_risk": served_item.predicted_risk,
                "arrival_period": served_item.arrival_period,
                "service_period": self.current_period,
                "simulated_waiting_age": waiting_age,
                "borough": served_item.borough,
                "true_resolution_hours": served_item.true_resolution_hours,
                "true_long_duration": served_item.true_long_duration,
            })
            self.total_slots_used += 1
        else:
            # Selected empty queue penalty
            reward -= self.config.reward_penalty_empty

        # Backlog penalty (risk-weighted)
        total_risk_backlog = sum(item.predicted_risk for q in self.queues for item in q)
        reward -= (self.config.reward_weight_backlog * (total_risk_backlog / 10.0))

        # Decrement capacity
        self.capacity_remaining -= 1

        # Check if period should advance
        total_waiting = sum(len(q) for q in self.queues)
        period_finished = (self.capacity_remaining <= 0) or (total_waiting == 0)

        if period_finished:
            # Record period snapshot
            high_risk_backlog = sum(1 for q in self.queues for item in q if item.predicted_risk >= 0.5)
            self.period_metrics.append({
                "period": self.current_period,
                "remaining_backlog": total_waiting,
                "high_risk_backlog": high_risk_backlog,
                "slots_used": self.config.capacity_per_period - max(0, self.capacity_remaining),
            })
            
            # Advance starvation counter for non-empty queues that were not served
            for q_idx in range(NUM_QUEUES):
                if len(self.queues[q_idx]) > 0:
                    self.starvation_tracker[q_idx] += 1
                    if self.starvation_tracker[q_idx] > self.max_starvation_observed[q_idx]:
                        self.max_starvation_observed[q_idx] = self.starvation_tracker[q_idx]

            # Starvation penalty if any queue waited too long
            max_starve = max(self.starvation_tracker)
            if max_starve >= 3:
                reward -= (self.config.reward_penalty_starve * (max_starve / 5.0))

            # Move to next period
            self.current_period += 1
            if self.current_period < self.horizon:
                self._ingest_current_period_arrivals()
                self.capacity_remaining = self.config.capacity_per_period

        done = self.current_period >= self.horizon

        if done:
            # Terminal backlog penalty
            rem_risk = sum(item.predicted_risk for q in self.queues for item in q)
            reward -= (self.config.reward_penalty_term * (rem_risk / 5.0))

        next_state = encode_state(self.queues, self.capacity_remaining)
        info = {
            "served": served_item is not None,
            "period": self.current_period,
            "capacity_remaining": self.capacity_remaining,
        }
        return next_state, reward, done, info

    def get_summary_metrics(self, policy_name: str) -> dict[str, Any]:
        """Compute end-of-simulation evaluation metrics."""
        total_handled = len(self.handled_items)
        remaining_backlog = sum(len(q) for q in self.queues)
        remaining_high_risk = sum(1 for q in self.queues for item in q if item.predicted_risk >= 0.5)
        
        # Waiting ages
        waiting_ages = [item["simulated_waiting_age"] for item in self.handled_items]
        mean_wait = float(np.mean(waiting_ages)) if waiting_ages else 0.0
        p95_wait = float(np.percentile(waiting_ages, 95)) if waiting_ages else 0.0
        
        # Queue distribution
        queue_counts = {q: 0 for q in FROZEN_QUEUES}
        for item in self.handled_items:
            queue_counts[item["queue_name"]] += 1
            
        queue_pcts = {
            q: round((queue_counts[q] / total_handled) * 100.0, 1) if total_handled > 0 else 0.0
            for q in FROZEN_QUEUES
        }

        # Retrospective outcome evaluation (where valid outcome labels exist)
        items_with_true_label = [item for item in self.handled_items if item["true_long_duration"] is not None]
        true_long_handled = sum(1 for item in items_with_true_label if item["true_long_duration"] == 1)
        pred_high_risk_handled = sum(1 for item in self.handled_items if item["predicted_risk"] >= 0.5)
        
        # All arrivals across horizon that had true labels
        all_arrivals = [item for batch in self.arrival_stream[:self.horizon] for item in batch]
        all_true_long = sum(1 for item in all_arrivals if item.true_long_duration == 1)
        
        true_long_capture_rate = (
            round((true_long_handled / all_true_long) * 100.0, 1) if all_true_long > 0 else 0.0
        )
        
        slot_utilization = (
            round((self.total_slots_used / self.total_slots_available) * 100.0, 1)
            if self.total_slots_available > 0 else 0.0
        )

        return {
            "policy": policy_name,
            "total_slots_available": self.total_slots_available,
            "slots_used": self.total_slots_used,
            "slot_utilization_pct": slot_utilization,
            "items_handled": total_handled,
            "remaining_backlog": remaining_backlog,
            "remaining_high_risk_backlog": remaining_high_risk,
            "mean_waiting_periods": round(mean_wait, 2),
            "p95_waiting_periods": round(p95_wait, 2),
            "max_starvation_periods": int(max(self.max_starvation_observed)),
            "queue_distribution_counts": queue_counts,
            "queue_distribution_pct": queue_pcts,
            "predicted_high_risk_handled": pred_high_risk_handled,
            "retrospective_true_long_handled": true_long_handled,
            "retrospective_true_long_cohort_total": all_true_long,
            "retrospective_true_long_capture_rate": true_long_capture_rate,
            "period_metrics": self.period_metrics,
        }


# ---------------- Tabular Q-Learning Agent ----------------

class QLearningAgent:
    """Tabular Q-learning agent for municipal complaint queue allocation."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.q_table: dict[str, list[float]] = {}
        self.epsilon = config.epsilon_start
        self.training_rewards: list[float] = []

    def get_q_values(self, state: str) -> list[float]:
        """Return Q-values for all 6 actions in the state (lazy initialization to 0.0)."""
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(NUM_QUEUES)]
        return self.q_table[state]

    def select_action(
        self,
        state: str,
        queues: list[list[SimulationComplaint]],
        greedy: bool = False,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Select action using epsilon-greedy (training) or greedy (evaluation).
        
        In tie-breaking or exploration, prefers non-empty queues when available.
        """
        non_empty = [idx for idx, q in enumerate(queues) if len(q) > 0]
        
        if rng is None:
            rng = np.random.default_rng(self.config.seed)

        # Exploration
        if not greedy and (rng.random() < self.epsilon):
            if non_empty:
                return int(rng.choice(non_empty))
            return int(rng.integers(0, NUM_QUEUES))

        # Greedy choice
        q_vals = self.get_q_values(state)
        
        if non_empty:
            # Restrict greedy maximization to non-empty queues to avoid useless idle actions
            best_q = max(q_vals[idx] for idx in non_empty)
            candidates = [idx for idx in non_empty if np.isclose(q_vals[idx], best_q)]
            return int(rng.choice(candidates))
        else:
            best_q = max(q_vals)
            candidates = [idx for idx in range(NUM_QUEUES) if np.isclose(q_vals[idx], best_q)]
            return int(rng.choice(candidates))

    def update(
        self,
        state: str,
        action: int,
        reward: float,
        next_state: str,
        done: bool,
    ) -> float:
        """Bellman equation update: Q(s, a) <- Q(s, a) + alpha [r + gamma max Q(s', a') - Q(s, a)]."""
        q_vals = self.get_q_values(state)
        current_q = q_vals[action]
        
        if done:
            target = reward
        else:
            next_q_vals = self.get_q_values(next_state)
            target = reward + self.config.gamma * max(next_q_vals)

        td_error = target - current_q
        q_vals[action] = current_q + self.config.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        """Decay exploration rate smoothly towards epsilon_min."""
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def to_dict(self) -> dict[str, Any]:
        """Serialize agent parameters and Q-table."""
        return {
            "num_states": len(self.q_table),
            "num_actions": NUM_QUEUES,
            "actions": FROZEN_QUEUES,
            "q_table": self.q_table,
            "config": asdict(self.config),
            "training_rewards": self.training_rewards,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QLearningAgent:
        """Deserialize agent from dictionary."""
        config_data = data.get("config", {})
        config = SimulationConfig(**{k: v for k, v in config_data.items() if hasattr(SimulationConfig, k)})
        agent = cls(config)
        agent.q_table = data.get("q_table", {})
        agent.training_rewards = data.get("training_rewards", [])
        return agent


# ---------------- Baseline Policies ----------------

def policy_fifo(queues: list[list[SimulationComplaint]], rng: np.random.Generator) -> int:
    """FIFO: Select queue with the oldest waiting complaint (lowest arrival_period)."""
    oldest_period = 999999
    best_queues: list[int] = []
    
    for q_idx, q in enumerate(queues):
        if not q:
            continue
        first_item = q[0]
        if first_item.arrival_period < oldest_period:
            oldest_period = first_item.arrival_period
            best_queues = [q_idx]
        elif first_item.arrival_period == oldest_period:
            best_queues.append(q_idx)
            
    if best_queues:
        return int(rng.choice(best_queues))
    return int(rng.integers(0, NUM_QUEUES))


def policy_risk_first(queues: list[list[SimulationComplaint]], rng: np.random.Generator) -> int:
    """Risk-First: Select queue with the highest predicted long-duration risk complaint."""
    highest_risk = -1.0
    best_queues: list[int] = []
    
    for q_idx, q in enumerate(queues):
        if not q:
            continue
        for item in q:
            if item.predicted_risk > highest_risk:
                highest_risk = item.predicted_risk
                best_queues = [q_idx]
            elif np.isclose(item.predicted_risk, highest_risk):
                if q_idx not in best_queues:
                    best_queues.append(q_idx)
                    
    if best_queues:
        return int(rng.choice(best_queues))
    return int(rng.integers(0, NUM_QUEUES))


def policy_risk_plus_wait(
    queues: list[list[SimulationComplaint]],
    current_period: int,
    rng: np.random.Generator,
    weight_risk: float = 0.6,
) -> int:
    """Risk-Plus-Wait: Balance predicted risk and normalized waiting age."""
    best_score = -999.0
    best_queues: list[int] = []
    
    for q_idx, q in enumerate(queues):
        if not q:
            continue
        head = q[0]
        wait_norm = min(current_period - head.arrival_period, 5) / 5.0
        score = (weight_risk * head.predicted_risk) + ((1.0 - weight_risk) * wait_norm)
        
        if score > best_score:
            best_score = score
            best_queues = [q_idx]
        elif np.isclose(score, best_score):
            best_queues.append(q_idx)
            
    if best_queues:
        return int(rng.choice(best_queues))
    return int(rng.integers(0, NUM_QUEUES))


class RoundRobinPolicy:
    """Round-Robin: Sequentially cycle through non-empty queues."""
    def __init__(self) -> None:
        self.last_idx = -1

    def select(self, queues: list[list[SimulationComplaint]], rng: np.random.Generator) -> int:
        non_empty = [idx for idx, q in enumerate(queues) if len(q) > 0]
        if not non_empty:
            return int(rng.integers(0, NUM_QUEUES))
            
        # Find next non-empty queue cyclically after last_idx
        for offset in range(1, NUM_QUEUES + 1):
            cand = (self.last_idx + offset) % NUM_QUEUES
            if cand in non_empty:
                self.last_idx = cand
                return cand
                
        self.last_idx = non_empty[0]
        return non_empty[0]


# ---------------- Data Preparation & Out-of-Fold Risk Scoring ----------------

_OOF_RISK_CACHE: dict[str, np.ndarray] = {}

def get_complaint_pool(
    split_name: str = "train",
    use_oof_for_training: bool = True,
    seed: int = 42,
) -> list[SimulationComplaint]:
    """Load, prepare, and pre-score complaints from specified temporal split.
    
    - 'train': 2023 records. Uses 5-fold OOF probabilities if use_oof_for_training=True.
    - 'val':   2024-H1 records. Uses frozen Phase 2 classifier.
    - 'test':  2024-H2 records. Uses frozen Phase 2 classifier.
    
    Guarantees: Zero post-resolution leakage during feature extraction.
    """
    df = load_dataset(DATA_PATH)
    train_df, val_df, test_df = split_temporal_data(df)
    
    if split_name == "train":
        target_df = train_df
    elif split_name == "val":
        target_df = val_df
    elif split_name == "test":
        target_df = test_df
    else:
        raise ValueError(f"Unknown split_name '{split_name}'. Choose 'train', 'val', or 'test'.")

    # Extract intake features
    X, y_reg, _ = prepare_feature_matrix(target_df)

    # Compute risk probabilities
    if split_name == "train" and use_oof_for_training:
        cache_key = f"oof_train_{seed}"
        if cache_key in _OOF_RISK_CACHE:
            probs = _OOF_RISK_CACHE[cache_key]
        else:
            logger.info("Computing 5-fold out-of-fold (OOF) risk probabilities for 2023 training partition...")
            probs = np.zeros(len(train_df))
            threshold = compute_training_threshold(
                train_df[train_df["closed_date"].notna() & (train_df["resolution_hours"] >= 0)]
            )
            _, _, y_cls_train = prepare_feature_matrix(train_df, threshold=threshold)
            eligible_mask = (train_df["closed_date"].notna() & (train_df["resolution_hours"] >= 0)).values

            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (t_idx, v_idx) in enumerate(kf.split(train_df)):
                fit_idx = [i for i in t_idx if eligible_mask[i]]
                pipe = Pipeline([
                    ("preprocessor", build_preprocessor(min_category_frequency=15)),
                    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
                ])
                pipe.fit(X.iloc[fit_idx], y_cls_train.iloc[fit_idx])
                probs[v_idx] = pipe.predict_proba(X.iloc[v_idx])[:, 1]

            _OOF_RISK_CACHE[cache_key] = probs
    else:
        _, clf, _ = load_models()
        probs = clf.predict_proba(X)[:, 1]

    # Build SimulationComplaint pool
    complaints: list[SimulationComplaint] = []
    threshold = LONG_DURATION_THRESHOLD_HOURS
    
    for idx, (_, row) in enumerate(target_df.iterrows()):
        q_name = map_complaint_to_queue(row["complaint_type"])
        q_idx = QUEUE_TO_IDX[q_name]
        
        # True historical labels for retrospective evaluation only
        res_h = float(row["resolution_hours"]) if pd.notna(row.get("resolution_hours")) else None
        true_long = None
        if res_h is not None and res_h >= 0:
            true_long = 1 if res_h > threshold else 0

        complaint = SimulationComplaint(
            complaint_id=int(row.get("unique_key", idx)),
            complaint_type=str(row["complaint_type"]),
            queue_name=q_name,
            queue_idx=q_idx,
            predicted_risk=float(probs[idx]),
            arrival_period=0,
            borough=str(row.get("borough", "Unspecified")),
            true_resolution_hours=res_h,
            true_long_duration=true_long,
        )
        complaints.append(complaint)

    return complaints


def build_arrival_stream(
    complaint_pool: list[SimulationComplaint],
    horizon: int,
    batch_size: int,
    seed: int,
) -> list[list[SimulationComplaint]]:
    """Generate deterministic sequence of incoming complaint batches across periods."""
    rng = np.random.default_rng(seed)
    total_needed = horizon * batch_size
    
    if len(complaint_pool) < total_needed:
        # Sample with replacement if pool is smaller than required horizon
        indices = rng.choice(len(complaint_pool), size=total_needed, replace=True)
    else:
        # Sample without replacement for authentic historical variety
        indices = rng.choice(len(complaint_pool), size=total_needed, replace=False)

    stream: list[list[SimulationComplaint]] = []
    for p in range(horizon):
        batch_indices = indices[p * batch_size : (p + 1) * batch_size]
        batch = [complaint_pool[i] for i in batch_indices]
        stream.append(batch)
        
    return stream


# ---------------- Training & Evaluation Pipelines ----------------

def train_q_learning_policy(
    config: SimulationConfig,
    complaint_pool: list[SimulationComplaint] | None = None,
) -> tuple[QLearningAgent, list[float]]:
    """Train tabular Q-learning agent on 2023 training partition using out-of-fold risk scores."""
    config.validate()
    
    if complaint_pool is None:
        complaint_pool = get_complaint_pool(
            split_name="train",
            use_oof_for_training=config.use_oof_for_training,
            seed=config.seed,
        )

    agent = QLearningAgent(config)
    training_rewards: list[float] = []

    for ep in range(config.training_episodes):
        ep_seed = config.seed + (ep * 17)
        stream = build_arrival_stream(
            complaint_pool,
            horizon=config.horizon_periods,
            batch_size=config.arrival_batch_size,
            seed=ep_seed,
        )
        env = CivicQueueEnv(stream, config)
        state = env.reset()
        ep_reward = 0.0
        done = False
        
        ep_rng = np.random.default_rng(ep_seed)

        while not done:
            action = agent.select_action(state, env.queues, greedy=False, rng=ep_rng)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state

        agent.decay_epsilon()
        training_rewards.append(round(ep_reward, 2))

    agent.training_rewards = training_rewards
    return agent, training_rewards


def evaluate_single_policy(
    policy_name: str,
    env: CivicQueueEnv,
    agent: QLearningAgent | None = None,
    seed: int = 42,
) -> tuple[dict[str, Any], float]:
    """Execute a single policy against an environment instance and return metrics and reward."""
    state = env.reset()
    done = False
    total_reward = 0.0
    rng = np.random.default_rng(seed)
    
    rr_policy = RoundRobinPolicy() if policy_name == "Round-Robin" else None

    while not done:
        if policy_name == "Q-Learning":
            assert agent is not None, "Agent required for Q-Learning evaluation"
            action = agent.select_action(state, env.queues, greedy=True, rng=rng)
        elif policy_name == "FIFO":
            action = policy_fifo(env.queues, rng)
        elif policy_name == "Risk-First":
            action = policy_risk_first(env.queues, rng)
        elif policy_name == "Risk-Plus-Wait":
            action = policy_risk_plus_wait(env.queues, env.current_period, rng)
        elif policy_name == "Round-Robin":
            assert rr_policy is not None
            action = rr_policy.select(env.queues, rng)
        else:
            raise ValueError(f"Unknown policy '{policy_name}'")

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    metrics = env.get_summary_metrics(policy_name)
    metrics["total_reward"] = round(total_reward, 2)
    return metrics, total_reward


def run_policy_comparison(
    config: SimulationConfig,
    eval_split: str = "val",
    trained_agent: QLearningAgent | None = None,
    eval_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run comprehensive, multi-seed comparison across Q-learning and all 4 baseline policies.
    
    Guarantees: Every policy faces the exact same arrival stream for each seed.
    """
    config.validate()
    
    if eval_seeds is None:
        eval_seeds = [config.seed + i for i in range(5)]

    # Load evaluation complaint pool
    pool = get_complaint_pool(split_name=eval_split, use_oof_for_training=False, seed=config.seed)

    # Train agent if not provided
    train_rewards: list[float] = []
    if trained_agent is None:
        trained_agent, train_rewards = train_q_learning_policy(config)
    else:
        train_rewards = trained_agent.training_rewards

    policy_names = ["Q-Learning", "FIFO", "Risk-First", "Risk-Plus-Wait", "Round-Robin"]
    results_by_policy: dict[str, list[dict[str, Any]]] = {p: [] for p in policy_names}

    # Evaluate across identical streams for each seed
    for s_idx, s in enumerate(eval_seeds):
        stream = build_arrival_stream(
            pool,
            horizon=config.horizon_periods,
            batch_size=config.arrival_batch_size,
            seed=s,
        )

        for p_name in policy_names:
            env = CivicQueueEnv(stream, config)
            metrics, _ = evaluate_single_policy(p_name, env, agent=trained_agent, seed=s)
            results_by_policy[p_name].append(metrics)

    # Aggregate statistics across seeds
    summary_rows: list[dict[str, Any]] = []
    
    for p_name in policy_names:
        runs = results_by_policy[p_name]
        rewards = [r["total_reward"] for r in runs]
        slots_used = [r["slots_used"] for r in runs]
        utilizations = [r["slot_utilization_pct"] for r in runs]
        handled = [r["items_handled"] for r in runs]
        rem_backlog = [r["remaining_backlog"] for r in runs]
        rem_high_risk = [r["remaining_high_risk_backlog"] for r in runs]
        mean_waits = [r["mean_waiting_periods"] for r in runs]
        p95_waits = [r["p95_waiting_periods"] for r in runs]
        starvations = [r["max_starvation_periods"] for r in runs]
        pred_hr_handled = [r["predicted_high_risk_handled"] for r in runs]
        true_long_handled = [r["retrospective_true_long_handled"] for r in runs]
        true_long_capture = [r["retrospective_true_long_capture_rate"] for r in runs]

        # Queue distribution average
        avg_q_dist = {
            q: round(float(np.mean([r["queue_distribution_pct"][q] for r in runs])), 1)
            for q in FROZEN_QUEUES
        }

        summary_rows.append({
            "policy": p_name,
            "mean_reward": round(float(np.mean(rewards)), 2),
            "std_reward": round(float(np.std(rewards)), 2),
            "mean_slots_used": round(float(np.mean(slots_used)), 1),
            "mean_slot_utilization_pct": round(float(np.mean(utilizations)), 1),
            "mean_items_handled": round(float(np.mean(handled)), 1),
            "mean_remaining_backlog": round(float(np.mean(rem_backlog)), 1),
            "mean_remaining_high_risk": round(float(np.mean(rem_high_risk)), 1),
            "mean_waiting_periods": round(float(np.mean(mean_waits)), 2),
            "p95_waiting_periods": round(float(np.mean(p95_waits)), 2),
            "mean_max_starvation": round(float(np.mean(starvations)), 1),
            "mean_pred_high_risk_handled": round(float(np.mean(pred_hr_handled)), 1),
            "mean_true_long_handled": round(float(np.mean(true_long_handled)), 1),
            "mean_true_long_capture_pct": round(float(np.mean(true_long_capture)), 1),
            "queue_distribution_pct": avg_q_dist,
        })

    # Trajectories from seed 0 for charting
    representative_trajectories: dict[str, Any] = {}
    for p_name in policy_names:
        representative_trajectories[p_name] = results_by_policy[p_name][0]["period_metrics"]

    comparison_df = pd.DataFrame(summary_rows)

    return {
        "evaluation_split": eval_split,
        "config": asdict(config),
        "evaluation_seeds": eval_seeds,
        "summary": summary_rows,
        "dataframe": comparison_df,
        "training_rewards": train_rewards,
        "representative_trajectories": representative_trajectories,
        "trained_agent": trained_agent,
    }


# ---------------- Serialization & Artifact Writers ----------------

def save_phase4_artifacts(comparison_result: dict[str, Any]) -> None:
    """Save trained policy, metadata, simulation metrics, CSV, and figures."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    agent: QLearningAgent = comparison_result["trained_agent"]
    config: dict[str, Any] = comparison_result["config"]
    summary: list[dict[str, Any]] = comparison_result["summary"]

    # 1. Save Q-learning policy table
    policy_path = MODELS_DIR / "q_learning_policy.json"
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(agent.to_dict(), f, indent=2)
    logger.info("Saved Q-learning policy to %s", policy_path)

    # 2. Save Q-learning metadata
    meta_path = MODELS_DIR / "q_learning_metadata.json"
    meta = {
        "project": "CivicResolve AI",
        "phase": "Phase 4: Reinforcement Learning Resource Allocation",
        "algorithm": "Tabular Q-Learning",
        "state_representation": "Discretized (highest_risk_queue, oldest_queue, capacity_tier, backlog_tier, risk_pressure)",
        "action_space": FROZEN_QUEUES,
        "action_space_size": NUM_QUEUES,
        "training_partition": "2023 (NYC 311 Profiling Cohort)",
        "validation_partition": "2024-H1",
        "test_partition": "2024-H2",
        "num_q_states_explored": len(agent.q_table),
        "config": config,
        "reward_formulation": "r = w_risk * pred_risk + w_wait * norm_wait - w_backlog * risk_backlog - penalties",
        "scientific_disclaimer": "Simulated capacity allocation experiment on NYC 311 rank-stratified sample. Not a deployed municipal SLA or causal reduction proof.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 3. Save simulation metrics JSON
    metrics_path = OUTPUTS_DIR / "phase4_simulation_metrics.json"
    metrics_export = {
        "metadata": meta,
        "evaluation_split": comparison_result["evaluation_split"],
        "evaluation_seeds": comparison_result["evaluation_seeds"],
        "summary": summary,
        "training_reward_summary": {
            "initial_reward": comparison_result["training_rewards"][0] if comparison_result["training_rewards"] else 0.0,
            "final_reward": comparison_result["training_rewards"][-1] if comparison_result["training_rewards"] else 0.0,
            "mean_reward": round(float(np.mean(comparison_result["training_rewards"])), 2) if comparison_result["training_rewards"] else 0.0,
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)

    # 4. Save policy comparison CSV
    csv_path = OUTPUTS_DIR / "phase4_policy_comparison.csv"
    flat_rows = []
    for s in summary:
        row = copy.deepcopy(s)
        # Flatten queue distribution
        q_dist = row.pop("queue_distribution_pct", {})
        for q_name, pct in q_dist.items():
            row[f"pct_{q_name}"] = pct
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    logger.info("Saved policy comparison CSV to %s", csv_path)

    # 5. Generate and save static Matplotlib charts for reports & visual evidence
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Chart 1: Policy Comparison Bar Chart
        plt.figure(figsize=(10, 5))
        policies = [s["policy"] for s in summary]
        rewards = [s["mean_reward"] for s in summary]
        stds = [s["std_reward"] for s in summary]
        bars = plt.bar(policies, rewards, yerr=stds, capsize=5, color=["#0284c7", "#94a3b8", "#e11d48", "#0f766e", "#f59e0b"])
        plt.title("Mean Policy Reward Across Seeds (Phase 4 Simulation)", fontsize=12, fontweight="bold")
        plt.ylabel("Reward Score")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(figures_dir / "phase4_policy_comparison.png", dpi=150)
        plt.close()

        # Chart 2: Backlog Trajectory Across Periods
        plt.figure(figsize=(10, 5))
        trajs = comparison_result["representative_trajectories"]
        colors = {"Q-Learning": "#0284c7", "FIFO": "#64748b", "Risk-First": "#e11d48", "Risk-Plus-Wait": "#0f766e", "Round-Robin": "#d97706"}
        for p_name, p_metrics in trajs.items():
            periods = [m["period"] for m in p_metrics]
            backlog = [m["remaining_backlog"] for m in p_metrics]
            plt.plot(periods, backlog, label=p_name, color=colors.get(p_name, "#000"), linewidth=2)
        plt.title("Simulated Backlog Trajectory by Planning Period", fontsize=12, fontweight="bold")
        plt.xlabel("Simulation Period (t)")
        plt.ylabel("Remaining Backlog (Complaints)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(figures_dir / "phase4_backlog_trajectories.png", dpi=150)
        plt.close()

        # Chart 3: Q-Learning Training Reward Curve
        if comparison_result["training_rewards"]:
            plt.figure(figsize=(10, 4.5))
            rewards = comparison_result["training_rewards"]
            plt.plot(rewards, alpha=0.35, color="#0284c7", label="Raw Episode Reward")
            # 10-episode rolling average
            if len(rewards) >= 5:
                rolling = pd.Series(rewards).rolling(window=10, min_periods=1).mean()
                plt.plot(rolling, color="#0369a1", linewidth=2.5, label="10-Episode Moving Avg")
            plt.title("Q-Learning Policy Convergence (Training Epochs)", fontsize=12, fontweight="bold")
            plt.xlabel("Training Episode")
            plt.ylabel("Accumulated Reward")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(figures_dir / "phase4_training_curve.png", dpi=150)
            plt.close()

        # Chart 4: Queue Allocation Share (Fairness)
        plt.figure(figsize=(11, 5))
        df_bars = pd.DataFrame([s["queue_distribution_pct"] for s in summary], index=policies)
        df_bars.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="Spectral")
        plt.title("Service Slot Distribution by Complaint Queue (%)", fontsize=12, fontweight="bold")
        plt.ylabel("Percentage of Handled Complaints")
        plt.xlabel("Policy")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(figures_dir / "phase4_queue_fairness.png", dpi=150)
        plt.close()

        logger.info("Saved static Phase 4 figures to %s", figures_dir)
    except Exception as e:
        logger.warning("Could not generate static figures: %s", e)
