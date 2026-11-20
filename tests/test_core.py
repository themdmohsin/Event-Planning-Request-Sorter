"""One test per milestone."""
from app import core


# Milestone 1: request types and dataset.
def test_dataset_covers_every_label():
    labels = {label for _, label in core.DATASET}
    assert labels == set(core.LABELS)
    assert all(text.strip() for text, _ in core.DATASET)
    assert core.to_training_text("Hi", "venue_inquiry").endswith("<venue_inquiry>")
