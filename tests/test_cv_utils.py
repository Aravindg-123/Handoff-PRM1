import numpy as np
import pandas as pd

from common.cv_utils import oof_scores


class ReversedClassEstimator:
    def fit(self, x, y):
        self.classes_ = np.array([1, 0])
        return self

    def predict_proba(self, x):
        return np.tile([0.8, 0.2], (len(x), 1))


def test_oof_scores_selects_probability_for_label_one_by_class_value():
    df = pd.DataFrame(
        {
            "task_id": np.repeat([f"task_{i}" for i in range(5)], 2),
            "label": [0, 1] * 5,
            "feature": np.arange(10),
        },
        index=np.arange(100, 110),
    )

    scores = oof_scores(df, ["feature"], ReversedClassEstimator, n_splits=5)

    assert len(scores) == len(df)
    np.testing.assert_array_equal(scores, np.full(len(df), 0.8))


def test_oof_scores_rejects_misaligned_training_mask():
    df = pd.DataFrame(
        {
            "task_id": np.repeat([f"task_{i}" for i in range(5)], 2),
            "label": [0, 1] * 5,
            "feature": np.arange(10),
        }
    )

    try:
        oof_scores(
            df,
            ["feature"],
            ReversedClassEstimator,
            n_splits=5,
            drop_from_train=np.array([False]),
        )
    except ValueError as exc:
        assert "one value per DataFrame row" in str(exc)
    else:
        raise AssertionError("Expected misaligned training mask to be rejected.")
