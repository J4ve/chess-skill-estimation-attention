//! Algorithm R reservoir sampling, mirroring `preprocess_lichess.py:147-154`.
//!
//! The whole point of this module is that it is a *line-for-line* transcription
//! of the production Python, not an independent implementation of "reservoir
//! sampling". The two behaviours that matter and that a from-scratch rewrite
//! typically gets subtly different:
//!
//! * `eligible` is incremented for every eligible game *before* the draw, so
//!   the draw is `randint(0, eligible - 1)` against the post-increment count.
//! * The RNG is consulted **only once the reservoir is already full**. While it
//!   is filling, no random numbers are drawn at all. Drawing during the fill
//!   phase (a common variant) desynchronises every subsequent draw.
//!
//! Because the draw depends only on the running eligible count, and never on
//! any game's contents, this reproduces the production sample exactly given the
//! same eligibility bitstream and seed. That is the same property
//! `preprocess_fast.py` relies on for its replay pass; here it is applied
//! inline, while the game is still in hand, so no replay pass is needed.

use crate::pyrandom::PyRandom;

pub struct Reservoir<T> {
    max_games: usize,
    rng: PyRandom,
    slots: Vec<T>,
    eligible: u64,
}

impl<T> Reservoir<T> {
    pub fn new(max_games: usize, seed: u64) -> Self {
        Reservoir {
            max_games,
            rng: PyRandom::new(seed),
            slots: Vec::with_capacity(max_games.min(1 << 16)),
            eligible: 0,
        }
    }

    /// Offer one **eligible** game. Must be called exactly once per game that
    /// passed both `is_eligible` and `has_full_clocks`, and never for any other
    /// game: the eligible counter drives the RNG stream.
    ///
    /// Returns `true` if the item is now held in the reservoir.
    pub fn offer(&mut self, item: T) -> bool {
        self.offer_with(|| item)
    }

    /// As [`Reservoir::offer`], but the item is only materialised if it is
    /// actually going to be stored.
    ///
    /// This matters at corpus scale. A month has on the order of half a million
    /// eligible games but the reservoir only ever *stores* about
    /// `max_games * (1 + ln(eligible / max_games))` of them, roughly 120k for a
    /// real month. Rendering PGN text for the other ~400k games just to throw it
    /// away would put the extraction cost back into the hot loop that this whole
    /// exercise exists to remove.
    ///
    /// The RNG draw happens before `make` is called and unconditionally once the
    /// reservoir is full, so laziness cannot perturb the random stream.
    pub fn offer_with<F: FnOnce() -> T>(&mut self, make: F) -> bool {
        self.eligible += 1;
        if self.slots.len() < self.max_games {
            self.slots.push(make());
            true
        } else {
            let j = self.rng.randint_below(self.eligible) as usize;
            if j < self.max_games {
                self.slots[j] = make();
                true
            } else {
                false
            }
        }
    }

    /// Number of eligible games seen so far (the Python `eligible` counter).
    pub fn eligible(&self) -> u64 {
        self.eligible
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The reservoir contents in slot order. This is the order
    /// `preprocess_lichess.py` writes `game_0000000.pkl`, `game_0000001.pkl`,
    /// ... in, so it is part of the equivalence contract, not an incidental
    /// detail.
    pub fn into_slots(self) -> Vec<T> {
        self.slots
    }

    pub fn slots(&self) -> &[T] {
        &self.slots
    }
}
