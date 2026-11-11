"""Journal local des requêtes /ask (SQLite, data/api_log.sqlite, gitignoré).

Champs : horodatage, request_id, question, statut, refus_raison, ids des passages cités,
latence, coût, prompt_version. Jamais la réponse complète ni les extraits du guide.

Le coût cumulé du jour (date locale du serveur) se lit dans ce journal, pour le garde-fou
de budget : il survit donc à un redémarrage de l'API.
"""

import datetime as dt
import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
create table if not exists requests (
    ts              text    not null,   -- horodatage ISO 8601, heure locale avec décalage
    day             text    not null,   -- date locale AAAA-MM-JJ (budget journalier)
    request_id      text    not null primary key,
    question        text    not null,
    status          text    not null,   -- statut du pipeline, ou UPSTREAM_ERROR
    refus_raison    text,
    cited_chunk_ids text    not null,   -- JSON : ids des passages cités
    latency_ms      real    not null,
    cost_usd        real    not null default 0,
    prompt_version  text
)
"""
COLUMNS = (
    "ts",
    "day",
    "request_id",
    "question",
    "status",
    "refus_raison",
    "cited_chunk_ids",
    "latency_ms",
    "cost_usd",
    "prompt_version",
)


class RequestLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Connexion courte : transaction validée (ou annulée) puis connexion fermée."""
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def record(
        self,
        *,
        request_id: str,
        question: str,
        status: str,
        refus_raison: str | None,
        cited_chunk_ids: Sequence[int],
        latency_ms: float,
        cost_usd: float | None,
        prompt_version: str | None,
        now: dt.datetime | None = None,
    ) -> None:
        moment = (now or dt.datetime.now()).astimezone()
        row = (
            moment.isoformat(timespec="seconds"),
            moment.date().isoformat(),
            request_id,
            question,
            status,
            refus_raison,
            json.dumps(list(cited_chunk_ids)),
            float(latency_ms),
            float(cost_usd or 0.0),
            prompt_version,
        )
        placeholders = ", ".join("?" for _ in COLUMNS)
        with self._lock, self._connect() as conn:
            conn.execute(
                f"insert into requests ({', '.join(COLUMNS)}) values ({placeholders})", row
            )

    def spent_on(self, day: dt.date) -> float:
        """Coût LLM cumulé d'une journée (date locale)."""
        with self._lock, self._connect() as conn:
            (total,) = conn.execute(
                "select coalesce(sum(cost_usd), 0) from requests where day = ?",
                (day.isoformat(),),
            ).fetchone()
        return float(total)

    def spent_today(self) -> float:
        return self.spent_on(dt.datetime.now().astimezone().date())
