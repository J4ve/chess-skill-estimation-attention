//! `clkscan` - scan, sample and extract Lichess `.pgn.zst` dumps in one pass.
//!
//! Two modes, sharing one scanning core:
//!
//! * **scan** (default, unchanged from 0.1.0): decide `is_eligible` +
//!   `has_full_clocks` per game, count them, optionally write a verdict byte per
//!   game. This is what `preprocess_fast.py` drives as its pass 1.
//!
//! * **select** (`--emit-pgn`): the same scan, but Algorithm R runs *inline*
//!   while each game is still in hand, and the selected games' PGN is written
//!   out. This collapses the old two-pass pipeline into one: there is no replay
//!   pass, and `python-chess` never has to `skip_game` its way across the ~99.97%
//!   of the month that is thrown away.
//!
//! The eligibility predicates are reimplementations of
//! `prototype/src/preprocess_lichess.py`:
//!
//!   is_eligible(game)              (preprocess_lichess.py:66-80)
//!   has_full_clocks(game, 100)     (preprocess_lichess.py:83-99)
//!
//! and the sampler is a transcription of `preprocess_lichess.py:147-154`. See
//! `analysis/rust-pass2-preprocessor.md` for the equivalence evidence and
//! `src/reservoir.rs` / `src/pyrandom.rs` for why each is a literal port rather
//! than an independent implementation.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use pgn_reader::{BufferedReader, RawComment, RawHeader, SanPlus, Skip, Visitor};

use clkscan::reservoir::Reservoir;

const DEFAULT_MAX_PLIES: u32 = 100;
const DEFAULT_MAX_GAMES: usize = 30_000;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_LOG_EVERY: u64 = 200_000;

// ---------------------------------------------------------------- predicates

/// Python `str.isdigit()` on an ASCII header value: non-empty, all digits.
///
/// Python's `isdigit()` also accepts non-ASCII digit codepoints (superscripts,
/// Devanagari digits, ...). Lichess Elo and TimeControl headers are ASCII, so
/// this narrowing cannot change a verdict on real dump data. Noted rather than
/// hidden, because it is the only intentional divergence in the port.
#[inline]
fn is_digits(v: &[u8]) -> bool {
    !v.is_empty() && v.iter().all(u8::is_ascii_digit)
}

/// Python `re` `\s` for ASCII input: `[ \t\n\r\f\v]`.
#[inline]
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0c | 0x0b)
}

/// Literal port of `re.search(r"\[%clk\s+([^\]]+)\]", comment)`.
///
/// The subtlety worth spelling out: `\s+` is greedy but backtracks, and
/// `[^\]]+` can itself match whitespace. So `[%clk  ]` (two spaces) DOES match,
/// with the group capturing the second space, while `[%clk ]` (one space) does
/// not. The rule that reproduces this exactly is: after the literal `[%clk`
/// there must be at least one whitespace byte, and then the first `]` at or
/// after that position must leave at least one byte before it.
///
/// `re.search` scans every start position, so a failed candidate must not stop
/// the search: the loop continues past it.
fn has_clk(bytes: &[u8]) -> bool {
    const PAT: &[u8] = b"[%clk";
    let n = bytes.len();
    let mut i = 0usize;
    while i + PAT.len() <= n {
        if bytes[i] == b'[' && &bytes[i..i + PAT.len()] == PAT {
            let p = i + PAT.len();
            if p < n && is_ws(bytes[p]) {
                let mut j = p + 1;
                while j < n && bytes[j] != b']' {
                    j += 1;
                }
                if j < n && j >= p + 2 {
                    return true;
                }
            }
        }
        i += 1;
    }
    false
}

// ---------------------------------------------------------------- game buffer

/// One mainline ply, captured verbatim for re-emission.
struct Ply {
    san: String,
    comment: Vec<u8>,
}

/// Everything needed to re-emit a game as PGN.
///
/// Headers are captured verbatim and in order, which is what makes the emitted
/// game a faithful input to `parse_game`: it reads `TimeControl`, `WhiteElo`,
/// `BlackElo`, `White`, `Black` and `Result`, and `game.board()` honours a
/// `FEN`/`SetUp` pair if one is present. Capturing only the six headers
/// `parse_game` names today would silently break the day it reads a seventh.
#[derive(Default)]
struct GameBuf {
    headers: Vec<(Vec<u8>, Vec<u8>)>,
    prelude_comment: Vec<u8>,
    plies: Vec<Ply>,
}

impl GameBuf {
    fn clear(&mut self) {
        self.headers.clear();
        self.prelude_comment.clear();
        self.plies.clear();
    }

    fn header(&self, key: &[u8]) -> Option<&[u8]> {
        self.headers
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_slice())
    }
}

/// Escape a PGN header value: backslash and double-quote are the only two
/// characters the format escapes. Embedded newlines would break the tag pair, so
/// they are folded to spaces.
fn push_escaped_header(out: &mut String, v: &[u8]) {
    for &b in v {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' | b'\r' => out.push(' '),
            _ => push_byte(out, b),
        }
    }
}

/// Push a raw byte as a char. Dump data is UTF-8; a lone invalid byte becomes
/// U+FFFD, exactly as the Python side's `errors="replace"` TextIOWrapper would
/// render it (`preprocess_lichess.py:58`), so the two paths agree even on
/// malformed input.
#[inline]
fn push_byte(out: &mut String, b: u8) {
    if b.is_ascii() {
        out.push(b as char);
    } else {
        out.push('\u{fffd}');
    }
}

fn push_bytes_utf8(out: &mut String, v: &[u8]) {
    match std::str::from_utf8(v) {
        Ok(s) => out.push_str(s),
        Err(_) => out.push_str(&String::from_utf8_lossy(v)),
    }
}

/// Render a buffered game back to PGN.
///
/// Returns the number of brace characters that had to be neutralised inside
/// comments. A brace inside a comment cannot occur in well-formed PGN (it would
/// have terminated the comment during parsing), so a nonzero count anywhere in a
/// run is reported rather than swallowed.
fn render_pgn(buf: &GameBuf, out: &mut String) -> u64 {
    let mut sanitised = 0u64;

    for (k, v) in &buf.headers {
        out.push('[');
        push_bytes_utf8(out, k);
        out.push_str(" \"");
        push_escaped_header(out, v);
        out.push_str("\"]\n");
    }
    out.push('\n');

    let mut line_len = 0usize;
    let mut push_token = |out: &mut String, tok: &str| {
        // The PGN export format wants lines under 80 columns. python-chess does
        // not care, but keeping the emitted file spec-shaped means it can be fed
        // to any other PGN tool without surprises.
        if line_len > 0 && line_len + 1 + tok.len() > 79 {
            out.push('\n');
            line_len = 0;
        } else if line_len > 0 {
            out.push(' ');
            line_len += 1;
        }
        out.push_str(tok);
        line_len += tok.len();
    };

    if !buf.prelude_comment.is_empty() {
        let mut c = String::new();
        sanitised += push_comment(&mut c, &buf.prelude_comment);
        push_token(out, &c);
    }

    for (i, ply) in buf.plies.iter().enumerate() {
        // Every black move gets an explicit "N..." prefix. It is more verbose
        // than the minimal export format, but it stays unambiguous when a
        // comment sits between the two halves of a move, which is the normal
        // case in clock-annotated Lichess data.
        let num = if i % 2 == 0 {
            format!("{}.", i / 2 + 1)
        } else {
            format!("{}...", i / 2 + 1)
        };
        push_token(out, &num);
        push_token(out, &ply.san);
        if !ply.comment.is_empty() {
            let mut c = String::new();
            sanitised += push_comment(&mut c, &ply.comment);
            push_token(out, &c);
        }
    }

    let result = buf.header(b"Result").unwrap_or(b"*");
    let mut r = String::new();
    push_bytes_utf8(&mut r, result);
    push_token(out, &r);

    out.push_str("\n\n");
    sanitised
}

fn push_comment(out: &mut String, comment: &[u8]) -> u64 {
    let mut sanitised = 0u64;
    out.push_str("{ ");
    for &b in comment {
        match b {
            b'{' | b'}' => {
                sanitised += 1;
                out.push(' ');
            }
            b'\n' | b'\r' => out.push(' '),
            _ => push_byte(out, b),
        }
    }
    out.push_str(" }");
    sanitised
}

// ---------------------------------------------------------------- visitor

struct ClkScan {
    max_plies: u32,
    capture: bool,

    // header predicate state (is_eligible)
    variant_ok: bool,
    event_rated: bool,
    white_elo_ok: bool,
    black_elo_ok: bool,
    tc_ok: bool,

    // movetext state (has_full_clocks)
    ply_index: u32,
    open_ply: bool,
    open_has_clock: bool,
    clock_failed: bool,

    // extraction state
    buf: GameBuf,
    scratch: String,
    reservoir: Option<Reservoir<String>>,
    brace_sanitised: u64,

    // counters
    scanned: u64,
    eligible: u64,
    verdicts: Option<BufWriter<File>>,

    // progress
    log_every: u64,
}

impl ClkScan {
    fn new(
        max_plies: u32,
        verdicts: Option<BufWriter<File>>,
        reservoir: Option<Reservoir<String>>,
        log_every: u64,
    ) -> Self {
        ClkScan {
            max_plies,
            capture: reservoir.is_some(),
            variant_ok: true,
            event_rated: false,
            white_elo_ok: false,
            black_elo_ok: false,
            tc_ok: false,
            ply_index: 0,
            open_ply: false,
            open_has_clock: false,
            clock_failed: false,
            buf: GameBuf::default(),
            scratch: String::with_capacity(8192),
            reservoir,
            brace_sanitised: 0,
            scanned: 0,
            eligible: 0,
            verdicts,
            log_every,
        }
    }

    #[inline]
    fn headers_ok(&self) -> bool {
        self.variant_ok && self.event_rated && self.white_elo_ok && self.black_elo_ok && self.tc_ok
    }

    /// Settle the ply whose comments we were collecting. `has_full_clocks`
    /// rejects the game the moment a within-cap ply carries no `[%clk]`.
    #[inline]
    fn close_open_ply(&mut self) {
        if self.open_ply {
            if !self.open_has_clock {
                self.clock_failed = true;
            }
            self.open_ply = false;
        }
    }

    /// True while this game could still be selected, i.e. while buffering its
    /// movetext is not wasted work.
    #[inline]
    fn still_live(&self) -> bool {
        self.capture && !self.clock_failed
    }
}

impl Visitor for ClkScan {
    type Result = ();

    fn begin_game(&mut self) {
        self.variant_ok = true;
        self.event_rated = false;
        self.white_elo_ok = false;
        self.black_elo_ok = false;
        self.tc_ok = false;
        self.ply_index = 0;
        self.open_ply = false;
        self.open_has_clock = false;
        self.clock_failed = false;
        if self.capture {
            self.buf.clear();
        }
    }

    fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
        let v = value.as_bytes();
        match key {
            b"Variant" => self.variant_ok = v == b"Standard",
            b"Event" => {
                self.event_rated = v.windows(5).any(|w| w == b"Rated".as_slice());
            }
            b"WhiteElo" => self.white_elo_ok = is_digits(v),
            b"BlackElo" => self.black_elo_ok = is_digits(v),
            b"TimeControl" => {
                // tc.split("+") must give exactly 2 all-digit parts. This
                // rejects correspondence ("-") and malformed controls.
                let mut parts = v.split(|&b| b == b'+');
                self.tc_ok = match (parts.next(), parts.next(), parts.next()) {
                    (Some(a), Some(b), None) => is_digits(a) && is_digits(b),
                    _ => false,
                };
            }
            _ => {}
        }
        if self.capture {
            self.buf.headers.push((key.to_vec(), v.to_vec()));
        }
    }

    fn end_headers(&mut self) -> Skip {
        // The single biggest win available: a game that fails the header
        // predicate can never be eligible, so skip its movetext entirely
        // instead of tokenizing thousands of SAN tokens to throw them away.
        Skip(!self.headers_ok())
    }

    fn san(&mut self, san: SanPlus) {
        self.close_open_ply();
        self.ply_index += 1;
        if self.ply_index <= self.max_plies && !self.clock_failed {
            self.open_ply = true;
            self.open_has_clock = false;
        }
        if self.still_live() {
            self.buf.plies.push(Ply {
                san: san.to_string(),
                comment: Vec::new(),
            });
        }
    }

    fn comment(&mut self, comment: RawComment<'_>) {
        // Comments before the first move attach to the game node in
        // python-chess, never to ply 1, so `open_ply` gates this correctly.
        // Multiple comments on one ply are concatenated by python-chess; here
        // any one of them carrying the tag is enough, which is the same answer.
        let bytes = comment.as_bytes();
        if self.open_ply && !self.open_has_clock && has_clk(bytes) {
            self.open_has_clock = true;
        }
        if self.still_live() {
            match self.buf.plies.last_mut() {
                Some(p) => {
                    if !p.comment.is_empty() {
                        p.comment.push(b' ');
                    }
                    p.comment.extend_from_slice(bytes);
                }
                None => {
                    if !self.buf.prelude_comment.is_empty() {
                        self.buf.prelude_comment.push(b' ');
                    }
                    self.buf.prelude_comment.extend_from_slice(bytes);
                }
            }
        }
    }

    fn begin_variation(&mut self) -> Skip {
        // `has_full_clocks` walks `node.variation(0)`, the mainline only.
        Skip(true)
    }

    fn end_game(&mut self) -> Self::Result {
        self.close_open_ply();
        let ok = self.headers_ok() && !self.clock_failed && self.ply_index > 0;
        self.scanned += 1;
        if ok {
            self.eligible += 1;
        }
        if let Some(w) = self.verdicts.as_mut() {
            let _ = w.write_all(if ok { b"1\n" } else { b"0\n" });
        }

        if ok {
            if let Some(res) = self.reservoir.as_mut() {
                // Algorithm R, inline. The rendering closure only runs if this
                // game actually takes a slot.
                let buf = &self.buf;
                let scratch = &mut self.scratch;
                let mut sanitised = 0u64;
                res.offer_with(|| {
                    scratch.clear();
                    sanitised = render_pgn(buf, scratch);
                    scratch.clone()
                });
                self.brace_sanitised += sanitised;
            }
        }

        if self.log_every > 0 && self.scanned % self.log_every == 0 {
            // Same shape as preprocess_lichess.py:140-146 so existing log
            // tailing and the `grep -o 'scanned=[0-9]*'` progress probe in
            // corpus_stream_parallel.sh:484 keep working unchanged. Goes to
            // stderr so stdout stays a clean machine-readable summary; the
            // pipeline merges both into one log file.
            let held = self.reservoir.as_ref().map(|r| r.len()).unwrap_or(0);
            eprintln!(
                "scanned={} eligible={} reservoir={}",
                self.scanned, self.eligible, held
            );
        }
    }
}

// ---------------------------------------------------------------- CLI

const HELP: &str = "\
clkscan - scan, sample and extract Lichess .pgn.zst dumps in one pass.

USAGE:
    clkscan --input <FILE.pgn.zst> [OPTIONS]

MODES:
    (default)             Scan only. Count games and how many pass the
                          scan-phase eligibility check preprocess_lichess.py
                          applies (is_eligible + has_full_clocks).

    --emit-pgn <PATH>     One-pass select mode. Scan, run Algorithm R reservoir
                          sampling inline, and write the selected games to PATH
                          as PGN, in reservoir-slot order. Equivalent to
                          preprocess_lichess.py's scan+sample phase, without the
                          second python-chess pass over the whole archive.

OPTIONS:
    --input <PATH>        Input .pgn.zst archive. Required.
    --max-plies <N>       Ply cap for the clock-completeness check.
                          [default: 100]
    --max-games <N>       Reservoir size in select mode. [default: 30000]
    --seed <N>            RNG seed for reservoir sampling. Must match the Python
                          pipeline's --seed to reproduce its sample.
                          [default: 42]
    --verdicts <PATH>     Write one line per game, 1=eligible 0=not. Used by
                          preprocess_fast.py's replay pass and by the
                          equivalence checker.
    --stats-json <PATH>   Write scanned/eligible/selected counts as JSON.
    --log-every <N>       Emit a progress line to stderr every N games.
                          0 disables. [default: 200000]
    --limit <N>           Stop after N games, for bounded equivalence runs.
    --allow-truncated     Proceed even if the archive ends mid-game. Refused by
                          default in select mode, because a partial month
                          silently yields a partial, non-uniform sample.
    -h, --help            Print this help.

EXIT STATUS:
    0  success
    1  runtime error (unreadable input, truncated archive in select mode, ...)
    2  bad usage

Scan mode prints a single summary line to stdout:
    scanned=N eligible=M elapsed_s=S games_per_sec=R truncated=BOOL
";

struct Args {
    input: PathBuf,
    verdicts: Option<PathBuf>,
    emit_pgn: Option<PathBuf>,
    stats_json: Option<PathBuf>,
    max_plies: u32,
    max_games: usize,
    seed: u64,
    limit: u64,
    log_every: u64,
    allow_truncated: bool,
}

fn usage_err(msg: &str) -> ! {
    eprintln!("clkscan: {msg}");
    eprintln!("Try 'clkscan --help' for more information.");
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut input: Option<PathBuf> = None;
    let mut verdicts: Option<PathBuf> = None;
    let mut emit_pgn: Option<PathBuf> = None;
    let mut stats_json: Option<PathBuf> = None;
    let mut max_plies = DEFAULT_MAX_PLIES;
    let mut max_games = DEFAULT_MAX_GAMES;
    let mut seed = DEFAULT_SEED;
    let mut limit = u64::MAX;
    let mut log_every = DEFAULT_LOG_EVERY;
    let mut allow_truncated = false;

    let mut args = env::args_os().skip(1);
    while let Some(a) = args.next() {
        let a = a.to_string_lossy().into_owned();
        let mut need = |what: &str| -> String {
            args.next()
                .unwrap_or_else(|| usage_err(&format!("{a} requires a {what}")))
                .to_string_lossy()
                .into_owned()
        };
        match a.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0)
            }
            "--input" => input = Some(need("path").into()),
            "--verdicts" => verdicts = Some(need("path").into()),
            "--emit-pgn" => emit_pgn = Some(need("path").into()),
            "--stats-json" => stats_json = Some(need("path").into()),
            "--allow-truncated" => allow_truncated = true,
            "--max-plies" | "--max-games" | "--seed" | "--limit" | "--log-every" => {
                let raw = need("number");
                let n: u64 = raw
                    .parse()
                    .unwrap_or_else(|_| usage_err(&format!("{a}: {raw:?} is not a number")));
                match a.as_str() {
                    "--max-plies" => {
                        max_plies = u32::try_from(n)
                            .unwrap_or_else(|_| usage_err("--max-plies is too large"))
                    }
                    "--max-games" => {
                        max_games = usize::try_from(n)
                            .unwrap_or_else(|_| usage_err("--max-games is too large"))
                    }
                    "--seed" => seed = n,
                    "--limit" => limit = n,
                    _ => log_every = n,
                }
            }
            other => usage_err(&format!("unrecognised argument {other:?}")),
        }
    }

    let input = input.unwrap_or_else(|| usage_err("--input is required"));
    if emit_pgn.is_some() && max_games == 0 {
        usage_err("--max-games must be at least 1 in select mode");
    }
    Args {
        input,
        verdicts,
        emit_pgn,
        stats_json,
        max_plies,
        max_games,
        seed,
        limit,
        log_every,
        allow_truncated,
    }
}

fn create(path: &Path, what: &str) -> Result<BufWriter<File>, String> {
    File::create(path)
        .map(|f| BufWriter::with_capacity(1 << 20, f))
        .map_err(|e| format!("cannot create {what} {}: {e}", path.display()))
}

fn run() -> Result<(), String> {
    let args = parse_args();

    if !args.input.exists() {
        return Err(format!("input not found: {}", args.input.display()));
    }
    let file = File::open(&args.input)
        .map_err(|e| format!("cannot open {}: {e}", args.input.display()))?;

    let mut dec = zstd::stream::read::Decoder::new(file).map_err(|e| {
        format!(
            "{} is not a readable zstd stream: {e}",
            args.input.display()
        )
    })?;
    // Lichess dumps are compressed with a large window; the Python side uses
    // max_window_size=2**31 for the same reason.
    dec.window_log_max(31)
        .map_err(|e| format!("cannot widen the zstd window: {e}"))?;

    let verdicts = match &args.verdicts {
        Some(p) => Some(create(p, "verdict file")?),
        None => None,
    };
    let mut pgn_out = match &args.emit_pgn {
        Some(p) => Some(create(p, "PGN output")?),
        None => None,
    };
    let select = pgn_out.is_some();
    let reservoir = select.then(|| Reservoir::<String>::new(args.max_games, args.seed));

    if select {
        eprintln!(
            "clkscan: select mode, reservoir={} seed={} max_plies={}",
            args.max_games, args.seed, args.max_plies
        );
    }

    let mut visitor = ClkScan::new(args.max_plies, verdicts, reservoir, args.log_every);
    let mut reader = BufferedReader::new(dec);

    let start = Instant::now();
    let mut truncated = false;
    loop {
        if visitor.scanned >= args.limit {
            break;
        }
        match reader.read_game(&mut visitor) {
            Ok(Some(())) => {}
            Ok(None) => break,
            Err(e) => {
                // A deliberately truncated sample ends mid-frame. That is an
                // expected benchmark condition, not a failure: report it and
                // use the games that were fully read.
                //
                // Failing on the very first game is a different animal. A file
                // that yields zero complete games is not a truncated archive,
                // it is the wrong file: a non-zstd input, a corrupt download, or
                // an HTML error page saved under a .zst name. Reporting that as
                // `truncated=true` with a success exit status sends the caller
                // hunting for a download that finished perfectly well.
                if visitor.scanned == 0 {
                    return Err(format!(
                        "{}: could not read a single complete game ({e}). This is a \
                         malformed, corrupt or non-zstd input rather than a truncated \
                         archive; check that the download completed and is really a \
                         Lichess .pgn.zst dump.",
                        args.input.display()
                    ));
                }
                eprintln!("note: input ended early ({e}); reporting games read so far");
                truncated = true;
                break;
            }
        }
    }
    let secs = start.elapsed().as_secs_f64();

    if let Some(w) = visitor.verdicts.as_mut() {
        w.flush().map_err(|e| format!("writing verdicts: {e}"))?;
    }

    // Carry preprocess_fast.py:87-91's refusal into the merged tool: a partial
    // month silently produces a partial, non-uniform sample, and the archive is
    // deleted after processing.
    if truncated && select && !args.allow_truncated {
        return Err(format!(
            "{} ends mid-game after {} games. Refusing to write a sample from a \
             truncated archive: the reservoir would be drawn from a prefix of the \
             month, not the month. Re-download it, or pass --allow-truncated if \
             you are deliberately benchmarking on a prefix.",
            args.input.display(),
            visitor.scanned
        ));
    }

    let mut selected = 0usize;
    if let (Some(out), Some(res)) = (pgn_out.as_mut(), visitor.reservoir.take()) {
        let slots = res.into_slots();
        selected = slots.len();
        for pgn in &slots {
            out.write_all(pgn.as_bytes())
                .map_err(|e| format!("writing PGN output: {e}"))?;
        }
        out.flush().map_err(|e| format!("writing PGN output: {e}"))?;

        if selected < args.max_games {
            eprintln!(
                "warning: only {} eligible games in the archive, fewer than the \
                 requested reservoir of {}; wrote all of them",
                selected, args.max_games
            );
        }
        if visitor.brace_sanitised > 0 {
            eprintln!(
                "warning: neutralised {} brace character(s) inside PGN comments; \
                 well-formed input should contain none",
                visitor.brace_sanitised
            );
        }
    }

    if let Some(p) = &args.stats_json {
        let mut f = File::create(p).map_err(|e| format!("cannot create {}: {e}", p.display()))?;
        write!(
            f,
            "{{\"scanned\": {}, \"eligible\": {}, \"selected\": {}, \"seed\": {}, \
             \"max_games\": {}, \"max_plies\": {}, \"truncated\": {}, \
             \"elapsed_s\": {:.3}}}\n",
            visitor.scanned,
            visitor.eligible,
            selected,
            args.seed,
            args.max_games,
            args.max_plies,
            truncated,
            secs
        )
        .map_err(|e| format!("writing {}: {e}", p.display()))?;
    }

    let rate = if secs > 0.0 {
        visitor.scanned as f64 / secs
    } else {
        0.0
    };
    // Unchanged stdout contract: preprocess_fast.py parses this line.
    println!(
        "scanned={} eligible={} elapsed_s={:.3} games_per_sec={:.0} truncated={}",
        visitor.scanned, visitor.eligible, secs, rate, truncated
    );
    if select {
        println!("selected={selected}");
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("clkscan: error: {msg}");
            ExitCode::FAILURE
        }
    }
}
