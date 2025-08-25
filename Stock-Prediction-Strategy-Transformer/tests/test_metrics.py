from src.evaluate.metrics import rmse, mae, r2, directional_accuracy
import numpy as np

def test_metrics_shapes():
    y = np.array([0.1, -0.2, 0.0])
    assert isinstance(rmse(y, y), float)
    assert isinstance(mae(y, y), float)
    assert isinstance(r2(y, y), float)
    assert isinstance(directional_accuracy(y, y), float)