import numpy as np
import torch

from suicide_detection.models.bilstm_attention import BiLSTMAttention, BiLSTMAttentionConfig
from suicide_detection.models.svm_baseline import SVMBaseline


def test_svm_pipeline_trains_on_tiny_data():
    X = np.array(
        [
            "I feel okay today",
            "I am in pain",
            "great day at work",
            "I want to give up",
            "happy times",
            "struggling a lot",
        ]
    )
    y = np.array([0, 1, 0, 1, 0, 1])
    model = SVMBaseline(grid_search=False)
    pipe = model.build()
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape[0] == X.shape[0]


def test_bilstm_forward_pass():
    cfg = BiLSTMAttentionConfig(vocab_size=100, embedding_dim=32, hidden_dim=16)
    model = BiLSTMAttention(cfg)
    ids = torch.randint(0, 100, (4, 10))
    attn = torch.ones_like(ids)
    logits = model(ids, attn)
    assert logits.shape == (4, 2)
