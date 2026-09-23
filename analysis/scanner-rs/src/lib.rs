//! `clkscan` internals, exposed as a library so the CPython-equivalence tests
//! can drive the RNG and the sampler directly instead of only through the CLI.

pub mod pyrandom;
pub mod reservoir;
