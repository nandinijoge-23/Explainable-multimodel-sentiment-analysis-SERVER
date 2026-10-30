from pathlib import Path
import wave
import numpy as np

root = Path(r"database\Test Audio")

for file in root.rglob("*.wav"):
    if file.stem.endswith("_loud"):
        continue

    output = file.with_name(file.stem + "_loud.wav")

    with wave.open(str(file), "rb") as wav:
        params = wav.getparams()
        raw = wav.readframes(wav.getnframes())

    if params.sampwidth != 2:
        print("Skipped:", file.name)
        continue

    audio = np.frombuffer(raw, dtype=np.int16)

    audio = np.clip(
        audio.astype(np.float32) * 2.5,
        -32768,
        32767
    ).astype(np.int16)

    with wave.open(str(output), "wb") as wav:
        wav.setparams(params)
        wav.writeframes(audio.tobytes())

    print("Created:", output)

print("All audio files processed.")