import numpy as np

import gate


class PositiveModel:
    def predict_proba(self, features):
        assert features["function_name_preserved"].iloc[0] == 1.0
        return np.array([[0.1, 0.9]])


def test_gate_forwards_entry_point_to_feature_extraction(monkeypatch):
    received = {}

    def fake_extract_features(handoff, reference_text, entry_point=None):
        received.update(
            handoff=handoff,
            reference_text=reference_text,
            entry_point=entry_point,
        )
        return {"function_name_preserved": 1.0}

    monkeypatch.setattr(gate, "extract_features", fake_extract_features)

    instance = gate.HandoffGate.__new__(gate.HandoffGate)
    instance.model = PositiveModel()
    instance.explainer = None
    instance.feature_cols = ["function_name_preserved"]
    instance.threshold = 0.7

    result = instance.check("handoff", "problem", entry_point="canonical_name")

    assert received == {
        "handoff": "handoff",
        "reference_text": "problem",
        "entry_point": "canonical_name",
    }
    assert result["pass"] is True
