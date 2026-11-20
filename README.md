[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/67wfp4237r/badge.svg)](https://msrit.ask.thili.ai/b/67wfp4237r)

# Event Planning Request Sorter

A laptop CLI that classifies a client request into one of six event-planning categories and appends it to a local review log. The trained GPT-2 adapter is included in `checkpoint/`.

## Run on Windows

Open PowerShell in the folder containing this README. Use Python 3.12 and Git.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.cli hello
.\.venv\Scripts\python.exe -m app.cli classify "Please supply a vegetarian buffet for dinner"
```

Expected output: `catering_request: Please supply a vegetarian buffet for dinner`. The first classification downloads the public GPT-2 base weights (about 550 MB). Subsequent runs reuse the cache. No API key or hosted service is required. On macOS or Linux create the environment with `python3.12 -m venv .venv` and use `.venv/bin/python`.

## Classify and review

Omit the request to enter several requests interactively. A blank line exits. Supply one main request per entry. The model always chooses one category.

```powershell
.\.venv\Scripts\python.exe -m app.cli classify
.\.venv\Scripts\python.exe -m app.cli classify "Can we change the date?" --output review.jsonl
Get-Content data/requests.jsonl
```

The default log is `data/requests.jsonl`. Each UTF-8 line contains the original request, predicted label and UTC timestamp. Read entries in arrival order to review and delegate them.

| Label | Planner handles |
|---|---|
| venue_inquiry | Space, availability, capacity and access |
| catering_request | Menus, drinks and dietary needs |
| decor_request | Flowers, lighting and table settings |
| entertainment_booking | Musicians, performers and activities |
| budget_question | Quotes, deposits, discounts and invoices |
| schedule_change | Rescheduling, delays and cancellations |

## Training and tests

`app/examples.py` contains 60 labeled training requests, ten per category. The tokenizer adds six special tokens such as `<venue_inquiry>` and the GPT-2 token embedding table is resized. The classifier pools the first four pretrained GPT-2 blocks over up to 96 input tokens. Hugging Face Trainer fits a residual adapter and six output vectors on the frozen features. The label with the highest score is saved. The compact checkpoint includes the trained parameters and label order.

```powershell
.\.venv\Scripts\python.exe -m app.cli train --epochs 15 --model-dir models/retrained
.\.venv\Scripts\python.exe -m app.cli classify "Could we rent the ballroom?" --model-dir models/retrained
.\.venv\Scripts\python.exe -m pytest -q
```

GitHub Actions runs pytest on pushes and pull requests using Python 3.12. Tests cover the dataset, special tokens, training, checkpoint loading, command-line predictions and append-only logging. Model tests download public weights on first use. Six unseen examples check the shipped checkpoint, one per category. This small check does not estimate accuracy on real client traffic.
