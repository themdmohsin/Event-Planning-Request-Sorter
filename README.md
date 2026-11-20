# Event Planning Request Sorter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/67wfp4237r/badge.svg)](https://ask.thili.ai/b/67wfp4237r)

The Event Planning Request Sorter is a tool for event planners to automatically sort incoming client requests. It uses a fine-tuned GPT-2 model to classify requests like "venue inquiry" or "catering request", streamlining task delegation and response prioritization. This project demonstrates fine-tuning GPT-2 for text classification and building an inference pipeline.

## What you are building

**Goal.** You will build an Event Planning Request Sorter that automatically classifies incoming client requests to help event planners efficiently delegate tasks and prioritize responses.

**Done when.** The project is done when your CLI tool can take a new event request, classify it into a predefined type using your fine-tuned GPT-2 model, and save the classified request to a file.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Define common event request types and create a small dataset of example requests.**  
      Hint: Start by brainstorming common request categories and then write a few example sentences for each to form your dataset.
- [ ] **2. Configure GPT-2 with an extended tokenizer and model for custom tokens.**  
      Hint: Refer to the drill's section on extending the tokenizer and model for special tokens like `<venue_inquiry>`.
- [ ] **3. Train the model on the example requests using the Hugging Face Trainer.**  
      Hint: Use the `Trainer` class from Hugging Face, similar to how GPT-2 was fine-tuned in the drill.
- [ ] **4. Build an interface (CLI) to input new requests and show their classification.**  
      Hint: Create a simple command-line interface to take user input and display the model's prediction.
- [ ] **5. Allow saving classified requests to a file for review.**  
      Hint: Implement file I/O to append the classified request and its label to a text file.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- (none)
