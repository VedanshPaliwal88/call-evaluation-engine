# Call Evaluation Project

Production-minded assignment solution for transcript analysis across three task areas:

- Q1: profanity detection
- Q2: privacy and compliance violation detection
- Q3: silence and overtalk metrics

## Stack

- Python 3.11+
- Streamlit
- Pydantic
- OpenAI SDK
- Plotly
- PyYAML
- python-dotenv
- pytest

## Project Layout

```text
app/
  streamlit_app.py
docs/
  architecture.md
src/
  call_evaluation/
tests/
data/
```

## Setup

1. Create a virtual environment with Python 3.11+.
2. Install dependencies:

```bash
pip install -e .[dev]
```

3. Configure environment variables:

```bash
copy .env.example .env
```

4. Update `.env` with a valid `OPENAI_API_KEY`.

## Run

```bash
streamlit run app/streamlit_app.py
```

## Test

```bash
pytest
```

If `data/labeled/annotations.csv` is available, it should be treated as the ground-truth validation set for Q1 and Q2 evaluation plus prompt regression checks.

## Deployment

The app is local-first and structured to remain compatible with Streamlit Community Cloud. The deployed environment needs the same environment variables as local execution.

## Notes

- The provided assignment data is JSON, but the app also supports YAML and ZIP uploads to match the assignment specification.
- `docs/architecture.md` contains the architecture rationale and diagram used by the implementation.
- A local-only technical report can be kept under `artifacts/local/` for review without committing it to Git.
