//! Desktop VRAM reserve state machine (Q-041).
//!
//! The engine holds `reserve_bytes` back from its own budget while the desktop is
//! healthy. When observed desktop free memory drops below the profile's release
//! threshold, the machine asks the engine to release memory; it re-acquires only
//! after `hold_steps` consecutive observations above the re-acquire threshold.
//! Every transition and every refusal is appended to a receipt log. An `UNKNOWN`
//! observation never causes a transition, and an Xid-dirty observation faults the
//! machine until an operator clears it.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReserveProfile {
    /// Bytes the engine gives back under pressure.
    pub release_bytes: u64,
    /// Desktop free bytes below which a release is requested.
    pub release_below_free: u64,
    /// Desktop free bytes above which re-acquiring is considered.
    pub reacquire_above_free: u64,
    /// Consecutive calm observations required before re-acquiring.
    pub hold_steps: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveState {
    /// Normal: engine at full budget, reserve kept free.
    Reserved,
    /// Release requested; waiting for the engine to confirm the bytes were returned.
    Releasing,
    /// Engine runs at `budget - release_bytes`.
    Released,
    /// Re-acquire requested; waiting for the engine's answer.
    Reacquiring,
    /// Driver fault observed; nothing changes until cleared.
    Faulted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Observation {
    pub step: u64,
    /// `None` is `UNKNOWN`.
    pub desktop_free: Option<u64>,
    /// `Some(true)` is `NO_XID`; `None` is `UNKNOWN`.
    pub driver_clean: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Release { bytes: u64 },
    Reacquire { bytes: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Transition {
        step: u64,
        from: ReserveState,
        to: ReserveState,
    },
    Unknown {
        step: u64,
    },
    /// The engine refused or failed a requested change; the reason is free text from the engine.
    Refused {
        step: u64,
        action: Action,
        reason: String,
    },
    /// The engine reported the release done but the observer did not see the bytes return.
    Unobserved {
        step: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveError {
    /// The call does not match the current state.
    WrongState(ReserveState),
}

#[derive(Debug)]
pub struct ReserveMachine {
    profile: ReserveProfile,
    state: ReserveState,
    calm: u32,
    last_step: u64,
    log: Vec<Event>,
}

impl ReserveMachine {
    pub fn new(profile: ReserveProfile) -> Self {
        Self {
            profile,
            state: ReserveState::Reserved,
            calm: 0,
            last_step: 0,
            log: Vec::new(),
        }
    }

    pub fn state(&self) -> ReserveState {
        self.state
    }
    pub fn log(&self) -> &[Event] {
        &self.log
    }

    fn go(&mut self, to: ReserveState) {
        self.log.push(Event::Transition {
            step: self.last_step,
            from: self.state,
            to,
        });
        self.state = to;
    }

    /// Feed one observation; returns the action the engine must perform, if any.
    pub fn observe(&mut self, obs: Observation) -> Option<Action> {
        self.last_step = obs.step;
        if obs.driver_clean == Some(false) {
            if self.state != ReserveState::Faulted {
                self.go(ReserveState::Faulted);
            }
            return None;
        }
        let Some(free) = obs.desktop_free else {
            self.log.push(Event::Unknown { step: obs.step });
            self.calm = 0;
            return None;
        };
        match self.state {
            ReserveState::Reserved if free < self.profile.release_below_free => {
                self.go(ReserveState::Releasing);
                Some(Action::Release {
                    bytes: self.profile.release_bytes,
                })
            }
            ReserveState::Released => {
                if free > self.profile.reacquire_above_free {
                    self.calm += 1;
                } else {
                    self.calm = 0;
                }
                if self.calm >= self.profile.hold_steps {
                    self.calm = 0;
                    self.go(ReserveState::Reacquiring);
                    Some(Action::Reacquire {
                        bytes: self.profile.release_bytes,
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Engine answer to a `Release`; `observed_returned` is the observer's view, not the engine's.
    pub fn confirm_release(
        &mut self,
        engine_ok: Result<(), String>,
        observed_returned: bool,
    ) -> Result<(), ReserveError> {
        if self.state != ReserveState::Releasing {
            return Err(ReserveError::WrongState(self.state));
        }
        match engine_ok {
            Ok(()) => {
                if !observed_returned {
                    self.log.push(Event::Unobserved {
                        step: self.last_step,
                    });
                }
                self.calm = 0;
                self.go(ReserveState::Released);
            }
            Err(reason) => {
                self.log.push(Event::Refused {
                    step: self.last_step,
                    action: Action::Release {
                        bytes: self.profile.release_bytes,
                    },
                    reason,
                });
                self.go(ReserveState::Reserved);
            }
        }
        Ok(())
    }

    /// Engine answer to a `Reacquire`; a refusal is typed, never a crash.
    pub fn confirm_reacquire(&mut self, engine_ok: Result<(), String>) -> Result<(), ReserveError> {
        if self.state != ReserveState::Reacquiring {
            return Err(ReserveError::WrongState(self.state));
        }
        match engine_ok {
            Ok(()) => self.go(ReserveState::Reserved),
            Err(reason) => {
                self.log.push(Event::Refused {
                    step: self.last_step,
                    action: Action::Reacquire {
                        bytes: self.profile.release_bytes,
                    },
                    reason,
                });
                self.go(ReserveState::Released);
            }
        }
        Ok(())
    }

    /// Operator clears a fault after a clean driver check; resumes in `Released` (safe side).
    pub fn clear_fault(&mut self, driver_clean: bool) -> Result<(), ReserveError> {
        if self.state != ReserveState::Faulted || !driver_clean {
            return Err(ReserveError::WrongState(self.state));
        }
        self.go(ReserveState::Released);
        Ok(())
    }

    /// Bytes the engine may use given its full budget.
    pub fn engine_budget(&self, full_budget: u64) -> u64 {
        match self.state {
            ReserveState::Reserved => full_budget,
            _ => full_budget.saturating_sub(self.profile.release_bytes),
        }
    }
}
