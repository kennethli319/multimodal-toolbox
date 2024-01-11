import gradio as gr
from transformers import pipeline
import numpy as np
import torch
from transformers import pipeline
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"

transcriber = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
  device=device,
)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

sample = ds[0]["audio"]

# we can also return timestamps for the predictions
prediction = transcriber(sample.copy(), batch_size=8, return_timestamps=True)["chunks"]

print(prediction)

pipe = pipeline(model="facebook/wav2vec2-base-960h")
# stride_length_s is a tuple of the left and right stride length.
# With only 1 number, both sides get the same stride, by default
# the stride_length on one side is 1/6th of the chunk_length_s
output = pipe(sample.copy(), chunk_length_s=10, stride_length_s=(4, 2))

print(output)

# transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")

# def transcribe(stream, new_chunk):
#     sr, y = new_chunk
#     y = y.astype(np.float32)
#     y /= np.max(np.abs(y))

#     if stream is not None:
#         stream = np.concatenate([stream, y])
#     else:
#         stream = y
#     return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]


# demo = gr.Interface(
#     transcribe,
#     ["state", gr.Audio(sources=["microphone"], streaming=True)],
#     ["state", "text"],
#     live=True,
# )

# demo.launch()