"""
Loop state — single source of truth for the run's history.

Tracks everything the loop has done so the agent, diagnosis, and (via
the event hooks) Person B's dashboard can always see the full picture.
Immutable within an iteration: a new IterationRecord is appended at the
end of each iteration, never mutated.

Public entry points
-------------------
IterationRecord : dataclass
    Snapshot of a single completed iteration. Fields:
        iteration: int
        model_name: str
        sources_pulled: list[str]      # all sources in working dataset this iter
        n_records: int                 # size of working dataset this iter
        accuracy: float
        f1: float
        train_accuracy: float
        diagnosis: str                 # one of the diagnosis constants
        diagnosis_reason: str          # human-readable reason string
        action_taken: str              # what the loop did in response

LoopState : dataclass
    Mutable container for the full run. Fields:
        iterations: list[IterationRecord]
        sources_pulled: list[str]      # hospital sources in working dataset so far
        models_tried: list[str]        # model names attempted so far
        current_model_name: str
        working_df: pd.DataFrame | None
        test_df: pd.DataFrame | None   # held fixed once set; never reassigned

    Methods:
        record(iter_record: IterationRecord) -> None
            Append a completed iteration to self.iterations and update
            sources_tried / models_tried if needed.

        sources_exhausted() -> bool
            True when len(sources_pulled) == len(SOURCES) from uci_client.

        summary() -> str
            Return a multi-line human-readable summary suitable for
            printing at the end of the run: best model, final accuracy,
            data sources used in order, benchmark comparison, iteration
            history table.

Implementation notes
-------------------
- working_df grows across iterations as new sources are merged in; test_df
  never changes after the first split.
- summary() should compare final accuracy against both benchmarks
  (78.7% baseline, 83.3% best-published) explicitly by name, not just
  print the raw number.
"""

import pandas as pd
from dataclasses import dataclass, field


@dataclass
class IterationRecord:
    iteration: int
    model_name: str
    sources_pulled: list[str]
    n_records: int
    accuracy: float
    f1: float
    train_accuracy: float
    diagnosis: str
    diagnosis_reason: str
    action_taken: str


@dataclass
class LoopState:
    iterations: list[IterationRecord] = field(default_factory=list)
    sources_pulled: list[str] = field(default_factory=list)
    models_tried: list[str] = field(default_factory=list)
    current_model_name: str = ""
    working_df: pd.DataFrame | None = None
    test_df: pd.DataFrame | None = None

    def record(self, iter_record: IterationRecord) -> None:
        """Append a completed iteration record and update tracking lists."""
        raise NotImplementedError

    def sources_exhausted(self) -> bool:
        """Return True when all four UCI hospital sources have been pulled."""
        raise NotImplementedError

    def summary(self) -> str:
        """Return a human-readable end-of-run summary with benchmark comparison."""
        raise NotImplementedError
