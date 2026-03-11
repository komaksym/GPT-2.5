import torch
import pytest

from pretrain.hellaswag import HellaSwagLoader, evaluate_hellaswag


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.num_rows = len(rows)

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError(f"Unsupported index: {item!r}")

        rows = self.rows[item]
        return {
            "ctx": [row["ctx"] for row in rows],
            "endings": [row["endings"] for row in rows],
            "label": [row["label"] for row in rows],
        }


class FakeTokenizer:
    _special_tokens = {"<|endoftext|>": 0}

    def encode_batch(self, texts):
        return [[ord(ch) for ch in text] for text in texts]


class DummyModel(torch.nn.Module):
    def forward(self, inputs):
        batch_size, seq_len = inputs.shape
        logits = torch.zeros(batch_size, seq_len, 8)
        return logits, None


def make_loader(batch_size=2, context_length=32):
    loader = HellaSwagLoader.__new__(HellaSwagLoader)
    loader.B = batch_size
    loader.T = context_length
    loader.dataset = FakeDataset(
        [
            {
                "ctx": "A",
                "endings": ["1", "2", "3", "4"],
                "label": "0",
            },
            {
                "ctx": "B",
                "endings": ["1", "2", "3", "4"],
                "label": "1",
            },
            {
                "ctx": "C",
                "endings": ["1", "2", "3", "4"],
                "label": "2",
            },
        ]
    )
    loader.tokenizer = FakeTokenizer()
    loader.cur_shard_pos = 0
    loader.n_examples = loader.dataset.num_rows
    loader.eos_token = 0
    return loader


def test_iter_batches_covers_full_dataset_including_tail_batch():
    loader = make_loader()

    batches = list(loader.iter_batches())

    assert [batch_labels.tolist() for _, batch_labels, _ in batches] == [[0, 1], [2]]
    assert [inputs.shape[0] for inputs, _, _ in batches] == [8, 4]


def test_evaluate_hellaswag_aggregates_accuracy_over_all_batches(monkeypatch):
    loader = make_loader()
    model = DummyModel()
    batch_results = iter([(2, 2), (0, 1)])

    def fake_compute(*args, **kwargs):
        return next(batch_results)

    monkeypatch.setattr("pretrain.hellaswag.compute_hellaswag_stats", fake_compute)

    score = evaluate_hellaswag(model, loader, torch.device("cpu"))

    assert score == pytest.approx(2 / 3)
    assert model.training is True
