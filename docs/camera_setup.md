# Camera Setup Notes

The virtual piano depends on stable webcam input and low-latency audio output.

## Before running

- Install dependencies with `pip install -r requirements.txt`.
- Use Python 3.8+.
- Confirm the webcam is not already being used by another application.
- Start in a bright room with a simple background.

## Run

```bash
python main.py
```

## Troubleshooting

- If the camera does not open, check the camera index in the code and OS privacy permissions.
- If hand tracking is unstable, increase lighting and keep both hands inside the frame.
- If audio is delayed, close other audio-heavy applications and restart the program.
- Generated `recordings/` files and `config.json` are local runtime artifacts and should stay out of Git.
