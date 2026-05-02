# Call Evaluation Project

## Project Overview

This project is a production-minded assignment solution for analyzing debt-collection call transcripts in batch. It supports three deliverables:

- `Q1`: profanity detection with both `regex` and `LLM prompting` approaches
- `Q2`: privacy and compliance violation detection with both `regex` and `LLM prompting` approaches
- `Q3`: silence and overtalk metrics with interactive Plotly visualizations

The application is built as a Streamlit app backed by a shared service layer so the same ingestion, detector, and metrics logic is used by both the UI and the test suite.

## Technology Stack

- Python `3.11+`
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
data/
  labeled/
docs/
  architecture.md
src/
  call_evaluation/
tests/
All_Conversations/
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "Call evaluation project"
```

### 2. Create the environment file

Create a `.env` file in the project root. The easiest path is to copy the example file:

```bash
copy .env.example .env
```

Then update `.env` with your values:

```env
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o
```

If `OPENAI_API_KEY` is missing, the app still runs, but the `LLM` approach is disabled in the UI and only `Pattern Matching` is available.

### 3. Install dependencies

Create and activate a Python `3.11+` environment, then install the project:

```bash
pip install -e .[dev]
```

## Run the Application

Start the Streamlit app with:

```bash
streamlit run app/streamlit_app.py
```

The UI supports:

- multiple `JSON` / `YAML` transcript uploads
- a single `ZIP` containing transcript files
- batch results by `call_id`
- evidence drill-down per call
- metrics visualizations for silence and overtalk

## Run the Tests

Run the full test suite with:

```bash
pytest
```

The repository includes:

- ingestion and validation tests
- regex detector adversarial tests
- prompt regression tests
- LLM response parsing tests
- metrics and visualization tests

## Deployment to Streamlit Community Cloud

1. Push the repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app from the GitHub repository.
4. Set the main file path to `app/streamlit_app.py`.
5. Add the required secrets in the Streamlit app settings:
   - `OPENAI_API_KEY`
   - `LLM_MODEL` with value `gpt-4o` or another supported model name
6. Deploy the app.

If the API key is omitted in deployment, the app still loads, but the `LLM` option is disabled and the UI shows a clear warning.

## Phase Summary

- `phase/1-foundation`
  - project scaffold, config, dependency setup, core models, architecture documentation, and prompt-template groundwork
- `phase/2-ingestion`
  - JSON/YAML/ZIP ingestion, canonical transcript normalization, validation, and special transcript tagging
- `phase/3-profanity`
  - regex and LLM profanity detectors, prompt regression coverage, and annotation-based evaluation
- `phase/4-compliance`
  - regex and LLM compliance detectors, voicemail/wrong-person/implied-verification handling, and annotation-based evaluation
- `phase/5-metrics`
  - interval-based silence/overtalk metrics, edge-case handling, and Plotly visualization helpers
- `phase/6-ui`
  - Streamlit batch analysis app, evidence drill-down, metrics dashboard, and graceful LLM degradation
- `phase/7-report`
  - final README and architecture documentation updates aligned to the shipped implementation

## Notes

- The provided assignment dataset is primarily JSON, but the system also supports `YAML` and `ZIP` input to match the assignment requirements.
- LLM prompts are versioned files under `src/call_evaluation/detectors/llm/prompts/`.
- The architecture reference for the final built system is documented in [docs/architecture.md](/C:/Users/Vedansh%20Paliwal/Desktop/Call%20evaluation%20project/docs/architecture.md).
