"""
Lichess game-by-ID client.

Fetches a game's PGN (with clock annotations) from the public Lichess export
API: https://lichess.org/api#tag/Games/operation/gameExport

Only one request is in flight at a time (serialized with a module-level
lock), so this prototype never hammers Lichess with concurrent requests.
"""

from __future__ import annotations

import asyncio
import re

import httpx

GAME_ID_RE = re.compile(r"^[A-Za-z0-9]{8}$")
# Matches a game ID inside a lichess.org URL, optionally followed by a color
# suffix (/black, /white), a slash, or a query string.
URL_ID_RE = re.compile(r"lichess\.org/(?:embed/)?([A-Za-z0-9]{8})(?:[/?#]|$)")

USER_AGENT = "RatingNet-Prototype/0.1 (+https://github.com/J4ve/RatingNet)"
EXPORT_URL_TEMPLATE = "https://lichess.org/game/export/{game_id}"
REQUEST_TIMEOUT_SECONDS = 10.0

_fetch_lock = asyncio.Lock()


class LichessError(Exception):
    """Raised for any Lichess game-ID or fetch failure, with an HTTP status to map to."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def parse_game_id(raw: str) -> str:
    """Extract an 8-character Lichess game ID from a bare ID or a full game URL."""
    candidate = raw.strip()
    if not candidate:
        raise LichessError(422, "Lichess game ID or URL must not be empty")

    url_match = URL_ID_RE.search(candidate)
    if url_match:
        return url_match.group(1)

    if GAME_ID_RE.match(candidate):
        return candidate

    raise LichessError(
        422,
        f"'{raw}' is not a valid Lichess game ID (8 alphanumeric characters) or game URL",
    )


async def fetch_game_pgn(game_id: str, client: httpx.AsyncClient | None = None) -> str:
    """Fetch a game's PGN, with clock annotations, from the Lichess export API.

    Raises LichessError with a status code of 404 (not found), 429 (rate
    limited), or 502 (unexpected upstream error/timeout).
    """
    url = EXPORT_URL_TEMPLATE.format(game_id=game_id)
    params = {"clocks": "true", "evals": "false", "opening": "false", "literate": "false"}
    headers = {"Accept": "application/x-chess-pgn", "User-Agent": USER_AGENT}

    async with _fetch_lock:
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)
        try:
            response = await client.get(url, params=params, headers=headers)
        except httpx.TimeoutException as exc:
            raise LichessError(502, "Timed out waiting for the Lichess API") from exc
        except httpx.HTTPError as exc:
            raise LichessError(502, f"Could not reach the Lichess API: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()

    if response.status_code == 404:
        raise LichessError(404, f"Lichess game '{game_id}' was not found")
    if response.status_code == 429:
        raise LichessError(429, "Lichess API rate limit exceeded; wait a moment and try again")
    if response.status_code != 200:
        raise LichessError(
            502, f"Lichess API returned an unexpected status ({response.status_code})"
        )

    pgn_text = response.text
    if not pgn_text.strip():
        raise LichessError(404, f"Lichess game '{game_id}' returned no PGN data")
    return pgn_text
