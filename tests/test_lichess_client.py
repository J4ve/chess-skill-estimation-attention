import asyncio

import httpx
import pytest

from lichess_client import LichessError, fetch_game_pgn, parse_game_id


def run(coro):
    """Run an async test body without pulling in a pytest-asyncio dependency."""
    return asyncio.run(coro)


def test_parse_bare_game_id():
    assert parse_game_id("abcd1234") == "abcd1234"


def test_parse_full_game_url():
    assert parse_game_id("https://lichess.org/abcd1234") == "abcd1234"


def test_parse_url_with_color_suffix():
    assert parse_game_id("https://lichess.org/abcd1234/black") == "abcd1234"


def test_parse_url_with_query_string():
    assert parse_game_id("https://lichess.org/abcd1234?any=thing") == "abcd1234"


def test_parse_embed_url():
    assert parse_game_id("https://lichess.org/embed/abcd1234") == "abcd1234"


def test_parse_rejects_wrong_length():
    with pytest.raises(LichessError) as excinfo:
        parse_game_id("short")
    assert excinfo.value.status_code == 422


def test_parse_rejects_empty_string():
    with pytest.raises(LichessError) as excinfo:
        parse_game_id("   ")
    assert excinfo.value.status_code == 422


def test_parse_rejects_garbage_url():
    with pytest.raises(LichessError) as excinfo:
        parse_game_id("https://example.com/not-a-lichess-link")
    assert excinfo.value.status_code == 422


def _client_returning(status_code: int, text: str = "") -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=text)

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


def test_fetch_success_returns_pgn_text():
    pgn = '[Event "Test"]\n\n1. e4 e5 *'

    async def body():
        async with _client_returning(200, pgn) as client:
            return await fetch_game_pgn("abcd1234", client=client)

    assert run(body()) == pgn


def test_fetch_not_found_maps_to_404():
    async def body():
        async with _client_returning(404) as client:
            await fetch_game_pgn("abcd1234", client=client)

    with pytest.raises(LichessError) as excinfo:
        run(body())
    assert excinfo.value.status_code == 404


def test_fetch_rate_limited_maps_to_429():
    async def body():
        async with _client_returning(429) as client:
            await fetch_game_pgn("abcd1234", client=client)

    with pytest.raises(LichessError) as excinfo:
        run(body())
    assert excinfo.value.status_code == 429


def test_fetch_unexpected_status_maps_to_502():
    async def body():
        async with _client_returning(500) as client:
            await fetch_game_pgn("abcd1234", client=client)

    with pytest.raises(LichessError) as excinfo:
        run(body())
    assert excinfo.value.status_code == 502


def test_fetch_empty_body_maps_to_404():
    async def body():
        async with _client_returning(200, "") as client:
            await fetch_game_pgn("abcd1234", client=client)

    with pytest.raises(LichessError) as excinfo:
        run(body())
    assert excinfo.value.status_code == 404
