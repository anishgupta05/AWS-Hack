"""
Model registry — the weak baseline and the pool of candidate replacements.

The deliberately weak starting model is a KNN with k=50 trained on
unscaled features. This is guaranteed to underperform on a ~300-record
dataset with mixed-scale features (cholesterol ~200, age ~50, binary
flags), so the correction loop fires reliably in the demo. This choice
is disclosed openly in the README.

The replacement pool, in rough order of expected performance on this
dataset, is: LogisticRegression (scaled), RandomForest, SVM (RBF),
GradientBoosting. No deep learning — the dataset is too small and
training time would kill the live demo.

Public entry points
-------------------
WEAK_BASELINE_NAME : str
    Name key for the deliberately weak starting model.

get_model(name: str) -> sklearn estimator
    Return a fresh (unfitted) sklearn estimator for the given name.
    Raises KeyError if the name is not in the registry.

next_model(tried: list[str]) -> str | None
    Return the name of the next candidate model to try, given the list
    of model names already attempted. Returns None when the full pool is
    exhausted (caller should treat this as "nothing left to switch to").
    Ordering is deterministic so the demo sequence is predictable.

MODEL_POOL : list[str]
    Ordered list of candidate model names, weak baseline first. Treat
    this as the canonical ordering for next_model().

Implementation notes
-------------------
- All models in the pool should be instantiated with reasonable
  defaults for this dataset (e.g. SVM with probability=True so
  per-class confidence is available to the diagnosis step).
- The weak baseline intentionally omits feature scaling — do not
  add a Pipeline wrapper for it. Scaled models should include a
  StandardScaler in a Pipeline so the harness doesn't need to
  know which models need scaling.
- random_state=42 on all stochastic models for reproducibility.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


WEAK_BASELINE_NAME = "knn_weak"

MODEL_POOL: list[str] = [
    "knn_weak",
    "logistic_regression",
    "random_forest",
    "svm_rbf",
    "gradient_boosting",
]

_REGISTRY: dict = {}  # populated below after class definitions


def get_model(name: str):
    """Return a fresh unfitted sklearn estimator for the given model name."""
    raise NotImplementedError


def next_model(tried: list[str]) -> str | None:
    """Return the next untried model name from MODEL_POOL, or None if exhausted."""
    raise NotImplementedError
