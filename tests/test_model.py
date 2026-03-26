"""
Smoke tests for the training pipeline.
These don't train a real model — they verify that the training
code runs without errors on tiny data.
"""
import pytest
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from src.training.train import SentimentDataset


class TestSentimentDataset:
    """Test the PyTorch dataset class."""

    @pytest.fixture
    def tokenizer(self):
        return BertTokenizer.from_pretrained("bert-base-uncased")

    def test_dataset_length(self, tokenizer):
        """Dataset length should match input data."""
        texts = ["hello world", "test text", "another one"]
        labels = [1, 0, 1]
        dataset = SentimentDataset(texts, labels, tokenizer, max_length=32)
        assert len(dataset) == 3

    def test_dataset_item_shape(self, tokenizer):
        """Each item should have correct tensor shapes."""
        texts = ["hello world"]
        labels = [1]
        dataset = SentimentDataset(texts, labels, tokenizer, max_length=32)
        item = dataset[0]

        assert item["input_ids"].shape == (32,)
        assert item["attention_mask"].shape == (32,)
        assert item["labels"].shape == ()
        assert item["labels"].item() == 1

    def test_dataset_label_dtype(self, tokenizer):
        """Labels should be long tensors for CrossEntropyLoss."""
        texts = ["test"]
        labels = [0]
        dataset = SentimentDataset(texts, labels, tokenizer, max_length=32)
        assert dataset[0]["labels"].dtype == torch.long


@pytest.mark.slow
class TestModelForwardPass:
    """Test that the model can do a forward pass (slow, needs model download)."""

    def test_forward_pass(self):
        """Model should produce logits of correct shape."""
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        model.eval()

        inputs = tokenizer("This is a test", return_tensors="pt", max_length=32,
                           truncation=True, padding="max_length")
        inputs["labels"] = torch.tensor([1])

        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs.logits.shape == (1, 2)
        assert outputs.loss is not None