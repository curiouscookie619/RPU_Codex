from __future__ import annotations

import os
import psycopg2

DEFAULT_DATABASE_URL = "postgresql://rpu:rpu@localhost:5432/rpu_app"

def get_conn():
    dsn = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    return psycopg2.connect(dsn)

def init_db():
    """Initialize DB objects required for logging and feedback.

    Safe to run on every app start; uses CREATE IF NOT EXISTS and small ALTER migrations.
    """
    with get_conn() as conn, conn.cursor() as cur:
        # Needed for gen_random_uuid()
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS event_log (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
              ts_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
              session_id TEXT NOT NULL,
              case_id TEXT,
              event_name TEXT NOT NULL,
              properties JSONB NOT NULL DEFAULT '{}'::jsonb
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
              case_id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              product_id TEXT NOT NULL,
              product_confidence NUMERIC,
              bi_date DATE NOT NULL,
              ptd DATE NOT NULL,
              rcd DATE NOT NULL,
              rpu_date DATE NOT NULL,
              mode TEXT NOT NULL,
              file_hash TEXT,
              extracted JSONB NOT NULL,
              outputs JSONB NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )

        # Lightweight migration for older deployments where updated_at didn't exist
        cur.execute("ALTER TABLE cases ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
              ts_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
              session_id TEXT NOT NULL,
              case_id TEXT,
              rating INT,
              comment TEXT
            );
            """
        )

        conn.commit()
