# Runtime Checklist

Use this checklist before testing the camera-based virtual piano system.

## Environment

- Confirm Python version and dependencies from `requirements.txt`.
- Verify webcam access at the target resolution.
- Check microphone or speaker output permissions if audio playback fails.
- Keep generated `config.json` local unless intentionally sharing a calibrated preset.

## Calibration

- Use stable, even lighting.
- Keep the hand area visible and avoid cluttered backgrounds.
- Confirm the piano key region aligns with the camera view.
- Save settings only after touch detection feels stable.

## Run Artifacts

Recordings and exported CSV or MIDI files should stay in `recordings/` or an external media folder. Commit only small examples or documentation needed to explain behavior.
