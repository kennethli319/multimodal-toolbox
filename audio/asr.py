import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, open_dict

models = {}

models['ctc'] = "nvidia/parakeet-ctc-1.1b"
models['rnnt'] = "nvidia/parakeet-rnnt-1.1b"

asr_model = nemo_asr.models.ASRModel.from_pretrained(models["ctc"])

decoding_cfg = asr_model.cfg.decoding
with open_dict(decoding_cfg):
    decoding_cfg.preserve_alignments = True
    decoding_cfg.compute_timestamps = True
    asr_model.change_decoding_strategy(decoding_cfg)

# specify flag `return_hypotheses=True``
hypotheses = asr_model.transcribe(["data/sample_22050.wav"], return_hypotheses=True)

# if hypotheses form a tuple (from RNNT), extract just "best" hypotheses
if type(hypotheses) == tuple and len(hypotheses) == 2:
    hypotheses = hypotheses[0]

timestamp_dict = hypotheses[0].timestep # extract timesteps from hypothesis of first (and only) audio file
print("Hypothesis contains following timestep information :", list(timestamp_dict.keys()))

# For a FastConformer model, you can display the word timestamps as follows:
# 80ms is duration of a timestep at output of the Conformer
time_stride = 8 * asr_model.cfg.preprocessor.window_stride

print(asr_model.cfg.preprocessor.window_stride)

word_timestamps = timestamp_dict['word']

for stamp in word_timestamps:
    start = stamp['start_offset'] * time_stride
    end = stamp['end_offset'] * time_stride
    word = stamp['char'] if 'char' in stamp else stamp['word']

    print(f"{start:0.2f} - {end:0.2f} - {word}")