use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use tokio::sync::{Notify, Semaphore, TryAcquireError};
use tracing::{debug, warn};

use crate::{config::WarmExecutionConfig, lifecycle::LifecycleRequest};

#[derive(Clone)]
pub struct RequestCancellation {
    inner: Arc<CancellationInner>,
}

struct CancellationInner {
    cancelled: AtomicBool,
    notify: Notify,
}

impl RequestCancellation {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(CancellationInner {
                cancelled: AtomicBool::new(false),
                notify: Notify::new(),
            }),
        }
    }

    pub fn cancel(&self) {
        if !self.inner.cancelled.swap(true, Ordering::SeqCst) {
            self.inner.notify.notify_waiters();
        }
    }

    pub async fn cancelled(&self) {
        let notified = self.inner.notify.notified();
        if self.is_cancelled() {
            return;
        }

        notified.await;
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }
}

impl Default for RequestCancellation {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WarmExecutionError {
    QueueFull,
    QueueTimeout,
}

impl WarmExecutionError {
    pub fn message(&self) -> &'static str {
        match self {
            Self::QueueFull => "warm execution queue is full",
            Self::QueueTimeout => "request did not start before the warm queue timeout",
        }
    }
}

#[derive(Clone)]
pub struct WarmExecutionScheduler {
    semaphore: Arc<Semaphore>,
    queue_state: Arc<Mutex<QueueState>>,
    config: WarmExecutionConfig,
}

struct QueueState {
    queued: usize,
}

impl WarmExecutionScheduler {
    pub fn new(config: WarmExecutionConfig) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(config.max_active_requests)),
            queue_state: Arc::new(Mutex::new(QueueState { queued: 0 })),
            config,
        }
    }

    pub async fn execute<T, F, Fut>(
        &self,
        _request: LifecycleRequest,
        operation: F,
    ) -> Result<T, WarmExecutionError>
    where
        T: Send,
        F: FnOnce(RequestCancellation) -> Fut + Send,
        Fut: std::future::Future<Output = T> + Send,
    {
        let permit = match self.semaphore.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::NoPermits) => {
                let queue_ticket = QueueTicket::reserve(
                    self.queue_state.clone(),
                    self.config.max_queued_requests,
                )?;
                let permit = tokio::time::timeout(
                    self.config.queue_timeout,
                    self.semaphore.clone().acquire_owned(),
                )
                .await;
                drop(queue_ticket);
                match permit {
                    Ok(Ok(permit)) => permit,
                    Ok(Err(_)) => panic!("warm execution semaphore should remain open"),
                    Err(_) => {
                        warn!("warm execution queue timeout");
                        return Err(WarmExecutionError::QueueTimeout);
                    }
                }
            }
            Err(TryAcquireError::Closed) => {
                panic!("warm execution semaphore should remain open")
            }
        };

        let cancellation = RequestCancellation::new();
        let cancel_on_drop = CancelOnDrop::new(cancellation.clone());
        debug!("warm execution slot acquired, starting operation");
        let output = operation(cancellation).await;
        cancel_on_drop.disarm();
        drop(permit);

        Ok(output)
    }
}

struct QueueTicket {
    queue_state: Arc<Mutex<QueueState>>,
    active: bool,
}

impl QueueTicket {
    fn reserve(
        queue_state: Arc<Mutex<QueueState>>,
        max_queued_requests: usize,
    ) -> Result<Self, WarmExecutionError> {
        let mut state = queue_state.lock().expect("queue state lock poisoned");
        if state.queued >= max_queued_requests {
            warn!(queued = state.queued, max = max_queued_requests, "warm execution queue full");
            return Err(WarmExecutionError::QueueFull);
        }

        state.queued += 1;
        drop(state);

        Ok(Self {
            queue_state,
            active: true,
        })
    }
}

impl Drop for QueueTicket {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        let mut state = self.queue_state.lock().expect("queue state lock poisoned");
        state.queued = state.queued.saturating_sub(1);
        self.active = false;
    }
}

struct CancelOnDrop {
    cancellation: RequestCancellation,
    armed: bool,
}

impl CancelOnDrop {
    fn new(cancellation: RequestCancellation) -> Self {
        Self {
            cancellation,
            armed: true,
        }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{future::pending, time::Duration};

    use tokio::{sync::oneshot, task::yield_now, time::timeout};

    use super::*;

    fn test_config() -> WarmExecutionConfig {
        WarmExecutionConfig {
            max_active_requests: 1,
            max_queued_requests: 1,
            queue_timeout: Duration::from_millis(25),
        }
    }

    #[tokio::test]
    async fn queued_request_times_out_with_overload_error() {
        let scheduler = WarmExecutionScheduler::new(test_config());
        let (release_tx, release_rx) = oneshot::channel::<()>();

        let active = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                scheduler
                    .execute(LifecycleRequest::Chat, |_| async move {
                        let _ = release_rx.await;
                    })
                    .await
                    .unwrap()
            }
        });

        yield_now().await;

        let error = scheduler
            .execute(LifecycleRequest::Embeddings, |_| async move {})
            .await
            .unwrap_err();
        assert_eq!(error, WarmExecutionError::QueueTimeout);

        let _ = release_tx.send(());
        active.await.unwrap();
    }

    #[tokio::test]
    async fn queue_limit_is_shared_across_request_types() {
        let scheduler = WarmExecutionScheduler::new(test_config());
        let (release_active_tx, release_active_rx) = oneshot::channel::<()>();
        let (release_queued_tx, release_queued_rx) = oneshot::channel::<()>();

        let active = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                scheduler
                    .execute(LifecycleRequest::Chat, |_| async move {
                        let _ = release_active_rx.await;
                    })
                    .await
                    .unwrap()
            }
        });

        let queued = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                scheduler
                    .execute(LifecycleRequest::Embeddings, |_| async move {
                        let _ = release_queued_rx.await;
                    })
                    .await
                    .unwrap()
            }
        });

        yield_now().await;
        yield_now().await;

        let error = scheduler
            .execute(LifecycleRequest::Chat, |_| async move {})
            .await
            .unwrap_err();
        assert_eq!(error, WarmExecutionError::QueueFull);

        let _ = release_active_tx.send(());
        let _ = release_queued_tx.send(());
        active.await.unwrap();
        queued.await.unwrap();
    }

    #[tokio::test]
    async fn aborting_queued_request_releases_queue_slot() {
        let scheduler = WarmExecutionScheduler::new(test_config());
        let (release_active_tx, release_active_rx) = oneshot::channel::<()>();

        let active = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                scheduler
                    .execute(LifecycleRequest::Chat, |_| async move {
                        let _ = release_active_rx.await;
                    })
                    .await
                    .unwrap()
            }
        });

        let queued = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                let _ = scheduler
                    .execute(LifecycleRequest::Embeddings, |_| async move {})
                    .await;
            }
        });

        yield_now().await;
        yield_now().await;
        queued.abort();
        let _ = queued.await;

        let replacement = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                scheduler
                    .execute(LifecycleRequest::Embeddings, |_| async move {})
                    .await
            }
        });

        yield_now().await;
        let _ = release_active_tx.send(());
        active.await.unwrap();
        replacement.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn aborting_active_request_cancels_work_and_releases_permit() {
        let scheduler = WarmExecutionScheduler::new(WarmExecutionConfig {
            max_active_requests: 1,
            max_queued_requests: 0,
            queue_timeout: Duration::from_millis(25),
        });
        let (cancellation_tx, cancellation_rx) = oneshot::channel::<RequestCancellation>();

        let active = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                let _ = scheduler
                    .execute(LifecycleRequest::Chat, move |cancellation| async move {
                        let _ = cancellation_tx.send(cancellation.clone());
                        pending::<()>().await;
                    })
                    .await;
            }
        });

        yield_now().await;
        let cancellation = cancellation_rx
            .await
            .expect("request should expose cancellation handle before abort");
        active.abort();
        let _ = active.await;

        timeout(Duration::from_secs(1), cancellation.cancelled())
            .await
            .expect("active work should be cancelled when the request task is dropped");

        scheduler
            .execute(LifecycleRequest::Embeddings, |_| async move {})
            .await
            .unwrap();
    }
}
