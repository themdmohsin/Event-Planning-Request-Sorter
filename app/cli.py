"""Command-line entry point: `python -m app.cli --help`."""
import argparse

from app import core


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="event-planning-request-sorter", description="Event planners use this tool to automatically sort incoming client requests (e.g., \"venue inquiry\", \"catering request\") for efficient task delegation and response prioritization.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    p = sub.add_parser("train", help="fine-tune GPT-2 on the example requests")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--model-dir", default=core.MODEL_DIR)
    p = sub.add_parser("classify", help="classify a request (or type them one per line if none given)")
    p.add_argument("text", nargs="?")
    p.add_argument("--model-dir", default=core.MODEL_DIR)
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Event Planning Request Sorter is ready. Open README.md for your milestones.")
    elif args.command == "train":
        loss = core.train(args.model_dir, epochs=args.epochs)
        print(f"Saved model to {args.model_dir} (training loss {loss:.3f})")
    elif args.command == "classify":
        tokenizer, model = core.load(args.model_dir)
        texts = [args.text] if args.text else _read_lines()
        for text in texts:
            print(f"{core.classify(text, tokenizer, model)}: {text}")


def _read_lines():
    print("Type a request and press Enter (empty line to quit).")
    while line := input("> ").strip():
        yield line


if __name__ == "__main__":
    main()
