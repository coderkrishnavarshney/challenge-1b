# Persona‑Driven Document Intelligence (Round 1B)

## Quick start (local)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.main --input sample_input
```

## Using Docker

```bash
docker build -t persona_doc_intel .
docker run --rm -v $PWD/sample_input:/data persona_doc_intel /data /output
```

The script writes `challenge1b_output.json` inside the specified `--output` directory (default `./results`).

All code runs **offline, CPU‑only** and respects the 1 GB model size ceiling (no neural language models are shipped).