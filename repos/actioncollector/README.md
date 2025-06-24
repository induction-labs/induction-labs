# actioncollector

Collect actions (keystrokes, mouse clicks, gestures) and their corresponding video frames, and write to remote bucket. Keep timestamps of everything.

## usage
```
uv run collect
```

## ignoring certain keystrokes

Add sensitive passwords etc into `.passwords`, with each password on a new line. Base64 encode and add like so:
```bash
 echo -n "your password(s)" | base64 >> .passwords
```