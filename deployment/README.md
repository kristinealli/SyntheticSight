# Deployment Interfaces

Both deployment interfaces import the shared `synthetic_sight` Python package. This is intentional: preprocessing, architecture, labels, and threshold logic should have one source of truth.

- `streamlit_app.py` — human-facing upload workflow.
- `api.py` — FastAPI endpoints for programmatic inference.
- `Dockerfile` — API container example.

The verified final checkpoint is **not included** in this package because the supplied archive contained only a legacy artifact. See `../models/README.md`.
