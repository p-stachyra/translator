import matplotlib.pyplot as plt
import numpy as np
import torch
import whisper


class TranscriptGenerator:
    def __init__(self, model_size) -> None:
        self.model_size = model_size
        self.model = whisper.load_model(self.model_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_sample(self, audio) -> np.array:
        return whisper.pad_or_trim(audio)
    
    def get_sample_spectrogram(self, audio):
        sample = self.get_sample(audio=audio)
        return whisper.log_mel_spectrogram(sample, n_mels=128).to(self.device)
    
    def detect_language(self, audio):
        spectrogram = self.get_sample_spectrogram(audio=audio)
        return self.model.detect_language(spectrogram)
    

if __name__ == "__main__":
    tg = TranscriptGenerator(model_size="large")
    #a = whisper.load_audio("sample.wav")
    #tg.get_sample_spectrogram()
    #plt.show()
