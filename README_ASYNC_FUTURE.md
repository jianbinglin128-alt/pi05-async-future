# Async future runtime

The current future-only runtime is documented in [README.md](README.md) and
[docs/future_runtime.md](docs/future_runtime.md).

It loads `Dream-LIN/pi05_libero_async_future`, predicts ten future actions,
executes the first five, and starts the next request after three executed actions
while the remaining two actions continue. New predictions already start after
the supplied context; do not discard their first two actions.

The legacy context evaluator and its overlapping-output schedule have been
removed. Historical training code remains available in earlier Git commits;
this snapshot supplies the validated inference runtime, rather than a recipe to
reproduce future-checkpoint training.
