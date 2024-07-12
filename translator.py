import matplotlib.pyplot as plt
import numpy as np
import torch
import whisper


AUDIO = whisper.load_audio(r"C:\Users\pstac\projects\translator\translator\sample.mp4")


class TranscriptGenerator:
    def __init__(self, model_name, device=None, number_mels: int = 80) -> None:
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        self.model = whisper.load_model(self.model_name, device=self.device)
        self.number_mels = number_mels

    def get_sample(self, audio) -> np.array:
        return whisper.pad_or_trim(audio)
    
    def get_sample_spectrogram(self, audio):
        sample = self.get_sample(audio=audio)
        return whisper.log_mel_spectrogram(sample, n_mels=self.number_mels).to(self.device)
    
    def detect_language(self, audio):
        sample = self.get_sample(audio=audio)
        spectrogram = whisper.log_mel_spectrogram(audio=sample, n_mels=self.number_mels).to(self.device)
        return self.model.detect_language(spectrogram)
    
    def __process_audio(self, audio, task_type: str = "transcribe"):
        _, language_probabilities = self.detect_language(audio=audio)
        most_probable_language = max(language_probabilities, key=language_probabilities.get)
        return self.model.transcribe(audio=audio, language=most_probable_language, task=task_type)
    
    def transcribe_audio(self, audio):
        return self.__process_audio(audio=audio, task_type="transcribe")
    
    def translate_to_english(self, audio):
        return self.__process_audio(audio=audio, task_type="translate")
    

if __name__ == "__main__":
    tg = TranscriptGenerator(model_name="small")
    # spec = tg.get_sample_spectrogram(audio=AUDIO).cpu()
    # plt.imshow(spec)
    # plt.show()

    _, probs = tg.detect_language(audio=AUDIO)
    print(probs)

    most_probable = max(probs, key=probs.get)

    print(tg.transcribe_audio(audio=AUDIO))

    print(tg.translate_to_english(audio=AUDIO))



    