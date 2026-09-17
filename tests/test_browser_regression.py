"""Browser regression suite: every bundled sample under every suspicion method.

Loads each of the 12 bundled samples in headless Chrome under each of the 4
methods (48 combinations) and asserts: no console error, the score and label
render, and the score matches its parity-checked expected value in
tests/regression/expected_sample_scores.json (to 1e-4). A second test switches
methods on one loaded game without reloading.

Opt-in, since it needs the rating checkpoint, Playwright and a Chrome binary:

    RATINGNET_BROWSER_REGRESSION=1 \\
    RATINGNET_CHECKPOINT=/path/to/best_model.pth \\
    RATINGNET_CHROME=/path/to/chrome \\
    pytest tests/test_browser_regression.py

It starts its own server on a free port (or set RATINGNET_REGRESSION_URL to use
one already running). RATINGNET_CHROME is optional when Playwright's own
Chromium is installed. After a deliberate model or cutoffs change, regenerate
the expected file with experiments/method_parity/build_regression_expected.py
and review its diff.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = json.loads((ROOT / "tests" / "regression" / "expected_sample_scores.json").read_text())
METHODS = ["s_att", "lgbm_a0g", "detector_a3g_seed0", "cnn_bilstm_a4"]
LABEL_SHORT = {"typical": "Typical", "unusual": "Unusual", "highly_unusual": "Highly unusual"}
TOLERANCE = EXPECTED["tolerance"]

pytestmark = pytest.mark.skipif(
    os.environ.get("RATINGNET_BROWSER_REGRESSION") != "1",
    reason="browser regression suite is opt-in: set RATINGNET_BROWSER_REGRESSION=1 (see module docstring)",
)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_healthy(url, proc, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if json.load(r).get("model_loaded"):
                    return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError("server did not become healthy in time")


@pytest.fixture(scope="module")
def base_url():
    existing = os.environ.get("RATINGNET_REGRESSION_URL")
    if existing:
        _wait_healthy(existing.rstrip("/"), None)
        yield existing.rstrip("/")
        return
    if not os.environ.get("RATINGNET_CHECKPOINT"):
        pytest.fail("set RATINGNET_CHECKPOINT (or RATINGNET_REGRESSION_URL) for the browser regression suite")
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "src" / "api.py")],
        env={**os.environ, "PORT": str(port)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_healthy(url, proc)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=30)


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        chrome = os.environ.get("RATINGNET_CHROME")
        b = p.chromium.launch(executable_path=chrome, headless=True) if chrome else p.chromium.launch(headless=True)
        yield b
        b.close()


def _open_page(browser, base_url, method):
    context = browser.new_context(viewport={"width": 1366, "height": 768})
    context.add_init_script(f"window.localStorage.setItem('ratingnet.suspicionMethod', {json.dumps(method)});")
    page = context.new_page()
    errors = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(base_url + "/")
    return context, page, errors


def _load_sample(page, sample_id):
    page.click("button[data-tab=samples]")
    page.click(f".sample-card[data-sample-id='{sample_id}'] .sample-card-main")
    page.wait_for_function(
        "document.getElementById('white-suspicion-value').dataset.score !== undefined", timeout=120_000
    )


def _assert_section_matches(page, method, expected):
    assert page.eval_on_selector("#suspicion-method-select", "e => e.value") == method
    assert page.eval_on_selector("#suspicion-section", "e => e.dataset.method") == method
    assert page.is_hidden("#suspicion-method-unavailable")
    for side in ("white", "black"):
        value = page.locator(f"#{side}-suspicion-value")
        score = float(value.get_attribute("data-score"))
        assert score == pytest.approx(expected[f"{side}_score"], abs=TOLERANCE), (method, side)
        digits = 1 if expected["scale"] == "rating_points" else 2
        assert value.text_content() == f"{score:.{digits}f}"
        chip = page.locator(f"#{side}-suspicion-label-chip")
        assert chip.is_visible()
        assert chip.text_content() == LABEL_SHORT[expected[f"{side}_label"]], (method, side)
    assert page.locator("#suspicion-method-caption").text_content().startswith("ROC-AUC ")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("sample", EXPECTED["samples"], ids=lambda s: s["id"])
def test_sample_renders_expected_score_under_method(browser, base_url, sample, method):
    context, page, errors = _open_page(browser, base_url, method)
    try:
        _load_sample(page, sample["id"])
        _assert_section_matches(page, method, sample["methods"][method])
        assert errors == []
    finally:
        context.close()


def test_switching_methods_rerenders_without_a_new_request(browser, base_url):
    sample = EXPECTED["samples"][0]
    context, page, errors = _open_page(browser, base_url, "detector_a3g_seed0")
    try:
        _load_sample(page, sample["id"])
        predict_calls = []
        page.on("request", lambda req: predict_calls.append(req.url) if "/predict/" in req.url else None)
        for method in METHODS + ["detector_a3g_seed0"]:
            page.select_option("#suspicion-method-select", method)
            _assert_section_matches(page, method, sample["methods"][method])
        assert predict_calls == []
        page.reload()
        assert page.eval_on_selector("#suspicion-method-select", "e => e.value") == "detector_a3g_seed0"
        assert errors == []
    finally:
        context.close()
