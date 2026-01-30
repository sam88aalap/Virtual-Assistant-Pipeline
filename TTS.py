import numpy as np
from pathlib import Path
from queue import Queue
from threading import Thread
import time
import soundfile as sf
import sounddevice as sd
from piper.voice import PiperVoice

MODEL_PATH = "models/en_US-kristin-medium.onnx"
voice = PiperVoice.load(MODEL_PATH)

AUDIO_OUTPUT_DIR = Path("audio_outputs")
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

def text_to_speech_file(text: str, end_silence_seconds: float = 2.0):
    audio_queue = Queue()
    sample_rate = None

    # Producer: generates audio and pushes into queue
    def synthesize():
        nonlocal sample_rate
        for chunk in voice.synthesize(text):
            if sample_rate is None:
                sample_rate = chunk.sample_rate

            audio_int16 = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_queue.put(audio_float32)

        # Add silence at the end
        silence = np.zeros(int(end_silence_seconds * sample_rate), dtype=np.float32)
        audio_queue.put(silence)

        # End signal
        audio_queue.put(None)

    # Start synth thread
    synth_thread = Thread(target=synthesize, daemon=True)
    synth_thread.start()

    # Collect audio
    audio_chunks = []

    while True:
        data = audio_queue.get()
        if data is None:
            break
        audio_chunks.append(data)

    # Concatenate all audio
    audio_float32 = np.concatenate(audio_chunks).astype(np.float32)

    # Output file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = AUDIO_OUTPUT_DIR / f"tts_{timestamp}.wav"

    sf.write(output_path, audio_float32, sample_rate, subtype="PCM_16")

    print(f"Reply Saved to {output_path}")
    return output_path

def text_to_speech_stream(text: str, end_silence_seconds: float = 2.0):
    audio_queue = Queue()
    sample_rate = None

    # Producer: generates audio and pushes into queue
    def synthesize():
        nonlocal sample_rate
        for chunk in voice.synthesize(text):
            if sample_rate is None:
                sample_rate = chunk.sample_rate

            audio_int16 = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_queue.put(audio_float32)

        # Add silence at the end
        silence = np.zeros(int(end_silence_seconds * sample_rate), dtype=np.float32)
        audio_queue.put(silence)

        # End signal
        audio_queue.put(None)

    # Consumer callback: fills output buffer
    buffer = np.empty((0,), dtype=np.float32)

    def callback(outdata, frames, time, status):
        nonlocal buffer

        if status:
            print("Stream status:", status)

        # fill buffer if not enough
        while len(buffer) < frames:
            data = audio_queue.get()
            if data is None:
                raise sd.CallbackStop()
            buffer = np.concatenate((buffer, data))

        # write exactly 'frames' samples
        outdata[:] = buffer[:frames].reshape(-1, 1)
        buffer = buffer[frames:]


    # Start synth thread
    synth_thread = Thread(target=synthesize, daemon=True)
    synth_thread.start()

    # wait for sample rate
    while sample_rate is None:
        pass

    # Start stream
    with sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback
    ):
        sd.sleep(int((len(text) / 10 + end_silence_seconds) * 1000))


if __name__ == "__main__":
    text_to_speech_stream("This is a TTS demo for you.")
