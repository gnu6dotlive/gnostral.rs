//! A small persistent thread pool for the grouped kernels.
//!
//! `run(tasks, f)` calls `f(i)` exactly once for every `i in 0..tasks`, on the
//! caller thread and the pool's workers, and returns only after every call has
//! finished. Work is claimed with an atomic counter, so *which* thread runs a
//! task is nondeterministic, but every kernel in this crate writes each output
//! element from exactly one task with a fixed reduction order: results never
//! depend on the thread count.
//!
//! A task that calls `run` on the pool it is running on executes the nested
//! call inline instead of deadlocking. A panicking task is re-raised on the
//! caller after every other task has finished.

use std::cell::Cell;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;

/// Environment variable that sizes [`Pool::global`].
pub const THREADS_ENV: &str = "GNOSTRAL_CPU_THREADS";

type Job = *const (dyn Fn(usize) + Sync);

struct JobSlot(Job);
// SAFETY: the pointee is `Sync` and outlives every use (see `Pool::run`).
unsafe impl Send for JobSlot {}

struct State {
    generation: u64,
    job: Option<JobSlot>,
    tasks: usize,
    active: usize,
    panic: Option<Box<dyn std::any::Any + Send>>,
    shutdown: bool,
}

struct Inner {
    state: Mutex<State>,
    start: Condvar,
    done: Condvar,
    next: AtomicUsize,
}

pub struct Pool {
    inner: Arc<Inner>,
    dispatch: Mutex<()>,
    workers: Vec<JoinHandle<()>>,
    threads: usize,
}

thread_local! {
    static IN_TASK: Cell<bool> = const { Cell::new(false) };
}

impl Pool {
    /// A pool with `threads` participants (the caller plus `threads - 1` workers).
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                generation: 0,
                job: None,
                tasks: 0,
                active: 0,
                panic: None,
                shutdown: false,
            }),
            start: Condvar::new(),
            done: Condvar::new(),
            next: AtomicUsize::new(0),
        });
        let workers = (1..threads)
            .map(|index| {
                let inner = inner.clone();
                std::thread::Builder::new()
                    .name(format!("gnostral-cpu-{index}"))
                    .spawn(move || worker(&inner))
                    .expect("spawn gnostral cpu worker")
            })
            .collect();
        Self {
            inner,
            dispatch: Mutex::new(()),
            workers,
            threads,
        }
    }

    /// Process-wide pool sized by [`THREADS_ENV`] or the available parallelism.
    pub fn global() -> &'static Pool {
        static GLOBAL: OnceLock<Pool> = OnceLock::new();
        GLOBAL.get_or_init(|| {
            let threads = std::env::var(THREADS_ENV)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
            Pool::new(threads)
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run `f(i)` for every `i in 0..tasks` and wait for all of them.
    pub fn run<F: Fn(usize) + Sync>(&self, tasks: usize, f: F) {
        if tasks == 0 {
            return;
        }
        if self.workers.is_empty() || tasks == 1 || IN_TASK.with(Cell::get) {
            (0..tasks).for_each(&f);
            return;
        }
        let _serial = self.dispatch.lock().unwrap_or_else(|e| e.into_inner());
        let job: &(dyn Fn(usize) + Sync) = &f;
        // SAFETY: lifetime erasure only; `Wait` below blocks until every worker
        // has dropped its use of the job, before `f` goes out of scope.
        let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(job) };
        {
            let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            self.inner.next.store(0, Ordering::SeqCst);
            state.job = Some(JobSlot(job));
            state.tasks = tasks;
            state.active = self.workers.len();
            state.generation = state.generation.wrapping_add(1);
            self.inner.start.notify_all();
        }
        let wait = Wait(&self.inner);
        let own = catch_unwind(AssertUnwindSafe(|| claim(&self.inner, tasks, &f)));
        drop(wait);
        let worker_panic = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .panic
            .take();
        if let Err(payload) = own {
            resume_unwind(payload);
        }
        if let Some(payload) = worker_panic {
            resume_unwind(payload);
        }
    }
}

struct Wait<'a>(&'a Inner);
impl Drop for Wait<'_> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        while state.active != 0 {
            state = self.0.done.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        state.job = None;
    }
}

struct InTask;
impl InTask {
    fn enter() -> Self {
        IN_TASK.with(|flag| flag.set(true));
        InTask
    }
}
impl Drop for InTask {
    fn drop(&mut self) {
        IN_TASK.with(|flag| flag.set(false));
    }
}

fn claim(inner: &Inner, tasks: usize, f: &(dyn Fn(usize) + Sync)) {
    let _in_task = InTask::enter();
    loop {
        let index = inner.next.fetch_add(1, Ordering::Relaxed);
        if index >= tasks {
            break;
        }
        f(index);
    }
}

fn worker(inner: &Inner) {
    let mut seen = 0u64;
    loop {
        let (job, tasks) = {
            let mut state = inner.state.lock().unwrap_or_else(|e| e.into_inner());
            while state.generation == seen && !state.shutdown {
                state = inner.start.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            if state.shutdown {
                return;
            }
            seen = state.generation;
            let job = state
                .job
                .as_ref()
                .map(|slot| slot.0)
                .expect("job published");
            (job, state.tasks)
        };
        // SAFETY: the dispatcher keeps the job alive until `active` reaches zero.
        let f = unsafe { &*job };
        let result = catch_unwind(AssertUnwindSafe(|| claim(inner, tasks, f)));
        let mut state = inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(payload) = result {
            state.panic.get_or_insert(payload);
        }
        state.active -= 1;
        if state.active == 0 {
            inner.done.notify_all();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            state.shutdown = true;
            self.inner.start.notify_all();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// A raw output pointer shared by tasks that write disjoint elements.
#[derive(Clone, Copy)]
pub(crate) struct SharedOut(*mut f32);
// SAFETY: every task writes a disjoint set of indices; see each kernel.
unsafe impl Send for SharedOut {}
unsafe impl Sync for SharedOut {}
impl SharedOut {
    pub(crate) fn new(out: &mut [f32]) -> Self {
        Self(out.as_mut_ptr())
    }
    /// # Safety
    /// `index` is in bounds and written by exactly one task.
    #[inline]
    pub(crate) unsafe fn write(self, index: usize, value: f32) {
        // SAFETY: forwarded to the caller.
        unsafe { *self.0.add(index) = value };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn every_task_runs_exactly_once() {
        let pool = Pool::new(4);
        for tasks in [0usize, 1, 3, 4, 17, 1000] {
            let hits: Vec<AtomicU64> = (0..tasks).map(|_| AtomicU64::new(0)).collect();
            pool.run(tasks, |i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
    }

    #[test]
    fn nested_run_executes_inline() {
        let pool = Pool::new(3);
        let total = AtomicU64::new(0);
        pool.run(4, |_| {
            pool.run(5, |_| {
                total.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(total.load(Ordering::Relaxed), 20);
    }

    #[test]
    fn worker_panic_is_reraised_after_all_tasks() {
        let pool = Pool::new(4);
        let finished = AtomicU64::new(0);
        let result = catch_unwind(AssertUnwindSafe(|| {
            pool.run(64, |i| {
                if i == 37 {
                    panic!("task 37");
                }
                finished.fetch_add(1, Ordering::Relaxed);
            })
        }));
        assert!(result.is_err());
        assert_eq!(finished.load(Ordering::Relaxed), 63);
        // The pool stays usable after a panic.
        let again = AtomicU64::new(0);
        pool.run(8, |_| {
            again.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(again.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn concurrent_dispatchers_serialise() {
        let pool = Arc::new(Pool::new(3));
        let total = Arc::new(AtomicU64::new(0));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let pool = pool.clone();
                let total = total.clone();
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        pool.run(7, |_| {
                            total.fetch_add(1, Ordering::Relaxed);
                        });
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(total.load(Ordering::Relaxed), 4 * 50 * 7);
    }
}
