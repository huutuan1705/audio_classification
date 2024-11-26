import torch 
import torchaudio

class AudioUtil():
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)