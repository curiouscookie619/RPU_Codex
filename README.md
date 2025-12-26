# RPU Calculators – Streamlit App (Product 1: Guaranteed Income STAR)

## What this is
Internal Streamlit web app that:
- accepts a Benefit Illustration (PDF),
- accepts PTD / Next Premium Due Date,
- derives RCD and RPU date (PTD + grace period),
- extracts numeric schedule values from the BI,
- computes Fully Paid vs Reduced Paid-Up (RPU) benefits (Guaranteed Income STAR),
- generates a neutral one-page output PDF,
- logs usage events + stores extracted/derived/computed data in Postgres.

## Setup (local)
1) Create a virtual environment and install requirements:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

2) Start Postgres:
```bash
docker compose up -d
```

3) Set env var (optional; defaults to local docker):
```bash
export DATABASE_URL="postgresql://rpu:rpu@localhost:5432/rpu_app"
```

4) Run the app:
```bash
streamlit run app.py
```

## Notes
- Uploaded PDFs are never stored. A SHA256 hash of the bytes may be stored for dedupe/debug.
- Names are used transiently for PDF personalization and are not persisted.
