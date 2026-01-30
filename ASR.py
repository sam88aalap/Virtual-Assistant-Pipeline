import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sounddevice as sd
import numpy as np
import queue
import librosa
from itertools import cycle
from faster_whisper import WhisperModel


MP3_DIR = "audio_inputs"
SAMPLE_RATE = 16000
CHANNELS = 1

mp3_files = cycle(
    sorted(
        os.path.join(MP3_DIR, f)
        for f in os.listdir(MP3_DIR)
        if f.endswith(".mp3")
    )
)

model = WhisperModel("base", device="cpu", compute_type="int8")


def is_speaking(buffer, threshold=0.01):
    return np.mean(np.abs(buffer)) > threshold

def listen_once() -> str:
    mp3_path = next(mp3_files)
    print(f"Listening...")

    # Load MP3 → mono float32 waveform
    audio, sr = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
    audio = audio.astype(np.float32)

    segments, _ = model.transcribe(
        audio,
        beam_size=1,
        vad_filter=False
    )

    text = " ".join(s.text.strip() for s in segments).strip()
    return text


def listen_once_mic(timeout=10, silence_duration=0.7) -> str:
    audio_queue = queue.Queue()
    buffer = np.empty((0, CHANNELS), dtype=np.float32)

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback,
    ):
        print("Listening...")
        silence_time = 0.0
        speaking = False

        while True:
            try:
                data = audio_queue.get(timeout=timeout)
            except queue.Empty:
                print("Timeout reached. No speech detected.")
                return ""

            buffer = np.concatenate((buffer, data))

            if len(buffer) > SAMPLE_RATE * 20:
                buffer = buffer[-SAMPLE_RATE * 20 :]

            duration = len(data) / SAMPLE_RATE

            if is_speaking(data):
                speaking = True
                silence_time = 0.0
            elif speaking:
                silence_time += duration

            if speaking and silence_time >= silence_duration:
                flat_buffer = buffer.flatten()

                segments, _ = model.transcribe(
                    flat_buffer,
                    beam_size=1,
                    vad_filter=False
                )
                text = " ".join([s.text.strip() for s in segments]).strip()
                return text


if __name__ == "__main__":
    text = listen_once()
    print("Transcribed:", text)
