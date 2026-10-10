use p3_maybe_rayon::prelude::join;

/// Prepare/consume share the device path; compute must only use host data.
/// One lookahead payload is retained, and oversized inputs drain the pipeline.
pub(super) fn map<P: Send, C: Send, R: Send>(
    count: usize,
    prefetch_bytes: usize,
    bytes: impl Fn(usize) -> usize,
    prepare: impl Fn(usize) -> P + Sync,
    compute: impl Fn(usize, P) -> C + Sync,
    consume: impl Fn(usize, C) -> R + Sync,
) -> Vec<R> {
    if prefetch_bytes == 0 || count < 2 {
        return (0..count)
            .map(|i| consume(i, compute(i, prepare(i))))
            .collect();
    }
    let mut results = Vec::with_capacity(count);
    let mut prepared = None;
    let mut pending = None;
    for i in 0..count {
        let input = prepared.take().unwrap_or_else(|| {
            if let Some((index, output)) = pending.take() {
                results.push(consume(index, output));
            }
            prepare(i)
        });
        let prefetch =
            i + 1 < count && bytes(i) <= prefetch_bytes && bytes(i + 1) <= prefetch_bytes;
        let previous = pending.take();
        let (output, (consumed, next)) = join(
            || compute(i, input),
            || {
                // Both closures may use nested Rayon work. Device operations
                // stay in this branch so a lease owner cannot recursively wait
                // for another partition requesting that same lease.
                let consumed = previous.map(|(index, output)| consume(index, output));
                let next = prefetch.then(|| prepare(i + 1));
                (consumed, next)
            },
        );
        results.extend(consumed);
        pending = Some((i, output));
        prepared = next;
    }
    if let Some((index, output)) = pending {
        results.push(consume(index, output));
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn ordering_and_oversized_inputs_preserve_bounded_payloads() {
        struct Payload(usize, Arc<Mutex<usize>>);
        impl Drop for Payload {
            fn drop(&mut self) {
                *self.1.lock().unwrap() -= self.0;
            }
        }
        for sizes in [vec![], vec![7], vec![4, 7, 11, 3, 2, 1]] {
            for limit in [0, 7] {
                let live = Arc::new(Mutex::new(0));
                let observed = Mutex::new(Vec::new());
                let result = map(
                    sizes.len(),
                    limit,
                    |i| sizes[i],
                    |i| {
                        let mut total = live.lock().unwrap();
                        *total += sizes[i];
                        assert!(*total <= sizes[i].max(limit * 2));
                        Payload(sizes[i], Arc::clone(&live))
                    },
                    |i, input| {
                        assert_eq!(input.0, sizes[i]);
                        i
                    },
                    |i, output| {
                        assert_eq!(i, output);
                        observed.lock().unwrap().push(i);
                        output
                    },
                );
                assert_eq!(result, (0..sizes.len()).collect::<Vec<_>>());
                assert_eq!(*observed.lock().unwrap(), result);
                assert_eq!(*live.lock().unwrap(), 0);
            }
        }
    }

    #[test]
    fn panic_joins_both_branches_and_releases_payloads() {
        struct Payload(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for Payload {
            fn drop(&mut self) {
                self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
        for failing_stage in 0..3 {
            let live = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let result = std::panic::catch_unwind(|| {
                map(
                    4,
                    1,
                    |_| 1,
                    |i| {
                        assert!(failing_stage != 0 || i != 1, "prepare failure");
                        live.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        Payload(Arc::clone(&live))
                    },
                    |i, input| {
                        assert!(failing_stage != 1 || i != 1, "compute failure");
                        input
                    },
                    |i, input| {
                        assert!(failing_stage != 2 || i != 1, "consume failure");
                        drop(input);
                    },
                )
            });
            assert!(result.is_err());
            assert_eq!(live.load(std::sync::atomic::Ordering::SeqCst), 0);
        }
    }
}
