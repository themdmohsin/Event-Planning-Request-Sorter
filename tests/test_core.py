"""One test per milestone."""
import math

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


# Milestone 3: training runs and saves a model we can load back.


def test_train_saves_a_loadable_model(tmp_path):
    loss = core.train(str(tmp_path), base_model=TINY, epochs=1)
    assert math.isfinite(loss)
    tokenizer, model = core.load(str(tmp_path))
    assert tokenizer.convert_tokens_to_ids("<venue_inquiry>") != tokenizer.unk_token_id


# Milestone 4: classify returns a known label and the CLI prints it.
def test_classify_and_cli(tmp_path, capsys):
    core.train(str(tmp_path), base_model=TINY, epochs=1)
    tokenizer, model = core.load(str(tmp_path))
    assert core.classify("Can we book the hall?", tokenizer, model) in core.LABELS

    from app.cli import main
    main(["classify", "Can we book the hall?", "--model-dir", str(tmp_path), "--output", str(tmp_path / "log.jsonl")])
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert any(out.startswith(label + ":") for label in core.LABELS)

