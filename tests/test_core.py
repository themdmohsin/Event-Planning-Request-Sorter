"""One test per milestone."""
from app import core


# Milestone 1: request types and dataset.
def test_dataset_covers_every_label():
    labels = {label for _, label in core.DATASET}
    assert labels == set(core.LABELS)
    assert all(text.strip() for text, _ in core.DATASET)
    assert core.to_training_text("Hi", "venue_inquiry").endswith("<venue_inquiry>")


# Milestone 2: label tokens are single new tokens the model has embeddings for.
TINY = "sshleifer/tiny-gpt2"  # a few MB, keeps tests fast


def test_tokenizer_and_model_know_label_tokens():
    tokenizer, model = core.build_model(TINY)
    for label in core.LABELS:
        ids = tokenizer.encode(core.label_token(label))
        assert len(ids) == 1
    assert model.get_input_embeddings().num_embeddings == len(tokenizer)
