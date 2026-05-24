# Calibration Notes

Use this checklist before testing the camera-based virtual piano.

## Camera Setup

- Use stable lighting with minimal shadows.
- Keep the hand and keyboard region fully visible.
- Confirm the webcam resolution and frame rate.
- Avoid reflective or cluttered backgrounds.

## Touch Detection

- Run a short calibration pass before recording.
- Check that each key region maps to the expected screen location.
- Verify false positives when hands hover above the keys.
- Save settings only after touch detection is stable.

## Debug Record

When calibration fails, record lighting, camera resolution, hand distance, FPS, and the generated `config.json` values.
