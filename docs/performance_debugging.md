# Performance Debugging

Use this checklist when the virtual piano feels slow, jittery, or inaccurate.

## Frame Loop

- Record webcam resolution and FPS.
- Check whether hand detection or UI drawing dominates frame time.
- Lower resolution temporarily to isolate camera throughput issues.
- Disable recording during baseline performance checks.

## Audio Latency

- Record mixer settings and active note count.
- Check whether overlapping notes create clipping or delay.
- Compare latency with and without recording enabled.

## Debug Record

For a performance issue, save the resolution, lighting setup, FPS range, CPU/GPU usage when available, and the generated configuration values.
