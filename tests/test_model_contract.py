from synthetic_sight.config import HEAD_DROPOUT, HEAD_HIDDEN_UNITS
from synthetic_sight.model import build_resnet50_binary


def test_final_classifier_head_shape():
    model = build_resnet50_binary()
    assert model.fc[0].in_features == 2048
    assert model.fc[0].out_features == HEAD_HIDDEN_UNITS
    assert model.fc[1].num_features == HEAD_HIDDEN_UNITS
    assert model.fc[3].p == HEAD_DROPOUT
    assert model.fc[4].out_features == 1
