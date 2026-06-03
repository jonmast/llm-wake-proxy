use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Default)]
pub struct Metrics {
    pub cold_starts: AtomicU64,
    pub warm_requests: AtomicU64,
    pub queue_full_rejections: AtomicU64,
    pub queue_timeouts: AtomicU64,
    pub wake_attempts: AtomicU64,
    pub wake_failures: AtomicU64,
    pub tunnel_drops: AtomicU64,
    pub embeddings_degraded: AtomicU64,
    pub forwarding_errors: AtomicU64,
    pub chat_requests: AtomicU64,
    pub embeddings_requests: AtomicU64,
}

#[derive(Debug, serde::Serialize)]
pub struct MetricsSnapshot {
    pub cold_starts: u64,
    pub warm_requests: u64,
    pub queue_full_rejections: u64,
    pub queue_timeouts: u64,
    pub wake_attempts: u64,
    pub wake_failures: u64,
    pub tunnel_drops: u64,
    pub embeddings_degraded: u64,
    pub forwarding_errors: u64,
    pub chat_requests: u64,
    pub embeddings_requests: u64,
}

impl Metrics {
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            cold_starts: self.cold_starts.load(Ordering::Relaxed),
            warm_requests: self.warm_requests.load(Ordering::Relaxed),
            queue_full_rejections: self.queue_full_rejections.load(Ordering::Relaxed),
            queue_timeouts: self.queue_timeouts.load(Ordering::Relaxed),
            wake_attempts: self.wake_attempts.load(Ordering::Relaxed),
            wake_failures: self.wake_failures.load(Ordering::Relaxed),
            tunnel_drops: self.tunnel_drops.load(Ordering::Relaxed),
            embeddings_degraded: self.embeddings_degraded.load(Ordering::Relaxed),
            forwarding_errors: self.forwarding_errors.load(Ordering::Relaxed),
            chat_requests: self.chat_requests.load(Ordering::Relaxed),
            embeddings_requests: self.embeddings_requests.load(Ordering::Relaxed),
        }
    }

    pub fn inc_cold_starts(&self) {
        self.cold_starts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_warm_requests(&self) {
        self.warm_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_queue_full(&self) {
        self.queue_full_rejections.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_queue_timeouts(&self) {
        self.queue_timeouts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_wake_attempts(&self) {
        self.wake_attempts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_wake_failures(&self) {
        self.wake_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_tunnel_drops(&self) {
        self.tunnel_drops.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_embeddings_degraded(&self) {
        self.embeddings_degraded.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_forwarding_errors(&self) {
        self.forwarding_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_chat_requests(&self) {
        self.chat_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_embeddings_requests(&self) {
        self.embeddings_requests.fetch_add(1, Ordering::Relaxed);
    }
}
