# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- **Environment**: the repo's own `.venv` may be unusable on some machines (e.g. unreadable interpreter symlinks on an NTFS-backed checkout). Use a venv outside the repo (`python3.12 -m venv ~/venvs/ratingnet-web`), CPU torch via `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt`. Point `RATINGNET_CHECKPOINT` at the frozen checkpoint rather than copying it; see README "Frozen weights".
- **Tests need no model.** `tests/` covers baseline resolution (`src/baseline.py`), critical-move ranking (`src/critical_moves.py`), Lichess ID/URL parsing and HTTP error mapping (`src/lichess_client.py`, mocked with `httpx.MockTransport`), and PGN parsing errors (`api._pgn_to_tensor_inputs`, which raises before any tensor/model code runs). None of these load the checkpoint, so `pytest tests/` runs fast with no `RATINGNET_CHECKPOINT` set.
- **chessboard.js sizes itself once, at construction, from its container's current rendered width.** If the container is inside a `hidden` element at that point, the board renders at zero size and never recovers on its own. Fix: call `board.resize()` right after un-hiding the container (see `src/static/app.js`, `renderResult()`).
- **`GET /` serves `index.html` directly (not through the `/static` mount)**, so every asset reference inside it (`<link>`, `<script src>`, the `pieceTheme` URL passed to chessboard.js) must be an absolute `/static/...` path, not a path relative to the page. Relative paths there resolve against `/`, not `/static/`, and 404 silently in the browser console. `app.js`'s own `import` statements are the one exception: those resolve relative to `/static/app.js`'s own URL, so `./vendor/...` is correct there.
- **Browser automation in this sandbox**: `chrome-devtools-axi` (and the `claude-in-chrome` MCP extension) may have no reachable Chrome/Chromium binary and fail with a generic "No page is currently selected" error regardless of retries. Workaround: `npx --yes @puppeteer/browsers install chrome@stable --path ~/.cache/puppeteer` to fetch a real Chrome for Testing binary, then drive it directly with Playwright (`chromium.launch(executable_path=..., headless=True)`) instead of the MCP browser tools. Do not run the installer without `--path`; it silently installs into the current working directory.
- **Lichess game export**: `GET https://lichess.org/game/export/{id}?clocks=true&evals=false` with `Accept: application/x-chess-pgn` and a descriptive `User-Agent`. 404 = not found, 429 = rate limited. An ongoing game has PGN header `Result "*"` and is delayed a few moves by Lichess itself.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
