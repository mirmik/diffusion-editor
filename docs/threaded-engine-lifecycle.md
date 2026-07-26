# Threaded engine lifecycle

The editor's in-process engines use a single-worker, event-queue contract. This
contract is deliberately small so that the UI controllers can keep the same
shape when an engine later moves to a process worker.

## Ownership and submission

- Each engine accepts at most one operation at a time.
- The submission check and worker ownership transfer happen under one lock.
- Model, pipeline, session, and GPU state stay owned by the engine. The worker
  may use that state, but the UI must not read or mutate it.
- A worker must not call Termin or other UI APIs.

## Results

- Workers publish immutable event objects through a thread-safe queue.
- The UI learns about completion, errors, and status only by polling that
  queue. It does not inspect worker-mutated result or error flags.
- A terminal event is enqueued before worker ownership is released. The queue
  remains busy until polling consumes that terminal event, and releases the
  slot before returning it. This prevents a back-to-back request from
  overtaking or being matched to the previous result while still letting a
  controller submit the next operation as it handles the event.
- Polling removes an event exactly once.

## Cancellation and shutdown

- Cancellation is cooperative: it sets a per-operation token. Operations that
  can cheaply check the token should stop promptly.
- A result completed after cancellation is converted to a cancellation error
  rather than delivered as successful output.
- Shutdown first stops new submissions, requests cancellation, and joins the
  worker for the configured timeout.
- Engine model or GPU state may be unloaded only after the worker has stopped.
  If the timeout expires, shutdown leaves that state intact so a live worker
  cannot touch an unloaded object.

CPython free-threading must not be treated as a reason to toggle the process
GIL or the garbage collector around an operation. Correctness comes from
explicit ownership, locks, cancellation tokens, and message passing.
