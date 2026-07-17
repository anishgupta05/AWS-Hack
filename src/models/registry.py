"""
Model registry — the weak baseline and the pool of candidate replacements.

The deliberately weak starting model is KNN with k=50 trained on raw,
unscaled features. On Cleveland-only (~242 training records after split),
k=50 means every prediction is a majority vote across ~21% of the training
set — effectively a smoothed majority-class classifier. The problem is
compounded by unscaled features: chol (~200) and trestbps (~130) dominate
Euclidean distance, making binary flags (sex, fbs, exang) nearly invisible.
Expected accuracy ~55–62%, well below the 78.7% logistic regression
baseline. This is disclosed in the README.

Candidate pool (MODEL_POOL order is the swap sequence used by next_model):
  knn_weak            — weak baseline, no pipeline (intentionally unscaled)
  logistic_regression — Pipeline(StandardScaler + LR), solid linear baseline
  random_forest       — no scaling needed, handles mixed features well
  svm_rbf             — Pipeline(StandardScaler + SVC), typically strong here
  gradient_boosting   — strongest single model, kept as last resort

Public entry points
-------------------
get_model(name) -> fresh unfitted sklearn estimator (or Pipeline)
next_model(tried) -> next name from MODEL_POOL not yet tried, or None
WEAK_BASELINE_NAME  -> "knn_weak"
MODEL_POOL          -> ordered list of all model names
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


WEAK_BASELINE_NAME: str = "knn_weak"

MODEL_POOL: list[str] = [
    "knn_weak",
    "logistic_regression",
    "random_forest",
    "svm_rbf",
    "gradient_boosting",
]

# Each entry is a zero-arg factory so get_model() always returns a fresh
# unfitted estimator — never hand a partially-fitted model to the loop.
_REGISTRY: dict[str, object] = {
    # No Pipeline wrapper: the unscaled features are the whole point.
    "knn_weak": lambda: KNeighborsClassifier(n_neighbors=50),

    # Pipeline: LR is sensitive to feature scale; StandardScaler is part
    # of the model, not a preprocessing step the harness has to manage.
    "logistic_regression": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),

    # Tree models are scale-invariant; no scaler needed.
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=42
    ),

    # SVM needs scaling; probability=True so per-class confidence is
    # available to the diagnosis step if needed later.
    "svm_rbf": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
    ]),

    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
}


def get_model(name: str):
    """Return a fresh unfitted estimator for the given model name.

    Raises KeyError if the name is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Available: {MODEL_POOL}"
        )
    return _REGISTRY[name]()


def next_model(tried: list[str]) -> str | None:
    """Return the next untried model name from MODEL_POOL, or None if exhausted.

    Iteration order is deterministic (MODEL_POOL order) so the demo
    sequence is predictable regardless of timing.
    """
    for name in MODEL_POOL:
        if name not in tried:
            return name
    return None
