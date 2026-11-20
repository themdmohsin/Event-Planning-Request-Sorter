from app.storage import append_request


def test_review_log_preserves_order_and_text(tmp_path):
    import json

    path = tmp_path / "review" / "requests.jsonl"
    for text in ('A "garden" venue', "Café\nmenu"):
        append_request(path, text, "venue_inquiry")
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["request"] for row in entries] == ['A "garden" venue', "Café\nmenu"]
    assert all(row["label"] == "venue_inquiry" for row in entries)


def test_shipped_checkpoint_on_unseen_requests():
    from app.core import classify, load

    tokenizer, model = load("checkpoint")
    cases = [
        ("Is the grand ballroom free next weekend?", "venue_inquiry"),
        ("Please supply a vegetarian buffet for dinner", "catering_request"),
        ("Could we have orchids on the tables?", "decor_request"),
        ("Please hire a pianist for the ceremony", "entertainment_booking"),
        ("How much is the deposit?", "budget_question"),
        ("Move our appointment to Wednesday", "schedule_change"),
    ]
    predictions = [classify(text, tokenizer, model) for text, _ in cases]
    assert predictions == [label for _, label in cases]
