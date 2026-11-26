//! Diagnostics: a timeline of a few frames. `VICTORY_TIMELINE=<file>` writes
//! one, from the first present after `VICTORY_TIMELINE_AT` seconds (60) on
//! and for `VICTORY_TIMELINE_FRAMES` frames (3);
//! `VICTORY_TIMELINE=<from s>[+<from s>..],<frames>,<file>` one for each of
//! the times (each in a file of its own, the second in its name): a Chrome trace (JSON, for
//! `chrome://tracing` or Perfetto) and beside it a text summary (`.txt`)
//! with the chain of work and waits each frame's present came out of.
//!
//! What is recorded while it runs, by every thread:
//! - spans of ours ([`span`]): the title's calls, the submitting thread's
//!   stages, the relay's, the pools' jobs, texture and streaming work;
//! - waits and the wake-ups that end them ([`hooks`]): the functions the
//!   title's threads wait with (its imports are bound to wrappers) and those
//!   our own code does (the entries of our own import table: every contended
//!   lock, condition and park of the standard library is a `WaitOnAddress`),
//!   each with the thread that woke it and where it waited;
//! - kernel calls on the address space that took long;
//! - the card: when each submission began and ended on it (timestamps).
//!
//! Without the variable nothing is wrapped and a span is one load of a
//! flag; a build without `diagnostics` has nothing of it.

#[cfg(all(windows, feature = "diagnostics"))]
mod hooks;
#[cfg(all(windows, feature = "diagnostics"))]
mod record;
#[cfg(all(windows, feature = "diagnostics"))]
mod report;

#[cfg(all(windows, feature = "diagnostics"))]
pub use hooks::import;
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) use record::{called, card_wait, on, polled, presented, sent, ticks, title_thread, told};

/// A span of ours: from where it is made until it is dropped. Every build
/// also counts it as a scope of the frame profiler (`victory_core::prof`,
/// named by the span's text); a diagnostics build also records it in the
/// timeline.
pub struct Span {
    #[cfg(all(windows, feature = "diagnostics"))]
    timeline: record::Span,
    /// Dropped before the next one begins (`next`).
    profiled: Option<victory_core::prof::Span>,
}

impl Span {
    /// The span ends here and another begins.
    #[inline(always)]
    pub fn next(&mut self, name: &'static str) {
        self.profiled = None;
        #[cfg(all(windows, feature = "diagnostics"))]
        self.timeline.next(name);
        self.profiled = Some(victory_core::prof::Span::named(name));
    }

    /// The span ends here.
    #[inline(always)]
    pub fn end(self) {}
}

/// A span named `name`.
#[inline(always)]
pub fn span(name: &'static str) -> Span {
    Span {
        #[cfg(all(windows, feature = "diagnostics"))]
        timeline: record::span(name),
        profiled: Some(victory_core::prof::Span::named(name)),
    }
}
