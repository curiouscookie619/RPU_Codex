from __future__ import annotations

import json
from typing import Any, Dict, Optional
from psycopg2.extras import Json
from .db import get_conn


def log_event(event_name: str, session_id: str, properties: Dict[str, Any], case_id: Optional[str] = None) -> None:
    # Always log to stdout as JSON (useful for hosting logs)
    payload = {
        "event_name": event_name,
        "session_id": session_id,
        "case_id": case_id,
        "properties": properties,
    }
    print(json.dumps(payload, default=str))

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO event_log(session_id, case_id, event_name, properties)
            VALUES (%s, %s, %s, %s::jsonb)
            """,
            (session_id, case_id, event_name, json.dumps(properties, default=str)),
        )
        conn.commit()
