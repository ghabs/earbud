# https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
# Extract the waveform
# Turn it into an embedding
# compare with cosine similarity
# Use a threshold to decide if it's the same speaker
# speechbrain uses a threshold of 0.25

from speechbrain.pretrained import SpeakerRecognition
import numpy as np
import torch

# Load the pretrained model
class Speaker():
    def __init__(self, voiceprint = None):
        self.voiceprint = voiceprint
        #TODO: save this into a cache dir
        self.model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    
    def verify(self, signal, signal2) -> tuple(float, bool):
        return self.model.verify_batch(signal, signal2)
    
    def create_voiceprint(self, signal: np.array) -> str:
        # stereo_audio = np.reshape(num_test, (-1, 2)).T  #  shape: (N,) → (2, N//2)
        #TODO add support for multiple samples
        mono_audio = np.reshape(signal, (-1, 1)).T
        tensor = torch.tensor(mono_audio)
        self.voiceprint = self.model.encode_batch(tensor)

    def check_voiceprint(self, signal: np.array) -> bool:
        mono_audio = np.reshape(signal, (-1, 1)).T
        tensor = torch.tensor(mono_audio)
        voiceprint = self.model.encode_batch(tensor)
        score, prediction = self.model.verify_batch(self.voiceprint, voiceprint)
        return prediction

