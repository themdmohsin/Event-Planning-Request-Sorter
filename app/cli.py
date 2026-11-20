"""Sort one request or enter several interactively."""
import argparse
from pathlib import Path

from app.storage import append_request


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify event planning requests with GPT-2.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check the installation")
    train = sub.add_parser("train", help="train a new adapter on the example requests")
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--model-dir", default="checkpoint")
    classify = sub.add_parser("classify", help="predict and save a request type")
    classify.add_argument("text", nargs="?")
    classify.add_argument("--model-dir", default="checkpoint")
    classify.add_argument("--output", type=Path, default=Path("data/requests.jsonl"))
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Event Planning Request Sorter is ready.")
        return
    from app import core
    try:
        if args.command == "train":
            loss = core.train(args.model_dir, epochs=args.epochs)
            print(f"Saved model to {args.model_dir} (training loss {loss:.3f})")
            return
        if args.text is not None and not args.text.strip():
            parser.error("Enter a non-empty request.")
        tokenizer, model = core.load(args.model_dir)
        texts = [args.text] if args.text is not None else read_requests()
        for text in texts:
            label = core.classify(text, tokenizer, model)
            append_request(args.output, text, label)
            print(f"{label}: {text}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Sorter error: {error}\n")


def read_requests():
    print("Enter one request per line. Press Enter on a blank line to finish.")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            return
        if not text:
            return
        yield text


if __name__ == "__main__":
    main()
