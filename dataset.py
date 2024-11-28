import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
from audio_util import AudioUtil

class SoundDataset(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # The relative path
        audio_file = self.data_path + self.df.loc[idx, 'relative_path']
        
        # Get the class id
        class_id = self.df.loc[idx, 'classID']
        aud = AudioUtil.open(audio_file)
        re_aud = AudioUtil.resample(aud, self.sr)
        re_chan = AudioUtil.rechannel(re_aud, self.channel)
        
        dur_aud = AudioUtil.pad_trunc(re_chan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        
        return aug_sgram, class_id
        
        