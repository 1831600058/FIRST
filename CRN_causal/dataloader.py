import random
import h5py
import numpy as np
import struct
import mmap
from torch.nn.utils.rnn import *
from torch.autograd.variable import *
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, args):
        self.args = args
        # noise
        self.f = open(args.noise_file, 'r+b')
        self.mm = mmap.mmap(self.f.fileno(), 0)
        self.len_noise = len(self.mm) // 4
        # speech
        self.f_wav = open(args.speech_file, 'r+b')
        self.mm_wav = mmap.mmap(self.f_wav.fileno(), 0)
        self.len_wav = len(self.mm_wav) // 4

        self.snr_lst = args.tr_snr
        self.snr_len = len(self.snr_lst)

        self.len = args.num_train
        self.speech_len = args.speech_len

    def __len__(self):
        return self.len

    def mix2signal(self, sig1, sig2, snr):
        alpha = np.sqrt(
            (np.sum(sig1 ** 2) / (np.sum(sig2 ** 2) + 1.0e-8)) / 10.0 ** (snr / 10.0) + 1.0e-8)
        return alpha

    def __getitem__(self, idx):
        len_speech = self.speech_len
        snr_idx = random.randint(0, self.snr_len - 1)
        Snr = self.snr_lst[snr_idx]
        # noise
        noise_loc = random.randint(0, self.len_noise - len_speech - 1)
        noise = np.array(list(struct.unpack('f' * len_speech, self.mm[noise_loc * 4:noise_loc * 4 + len_speech * 4])))
        # speech
        speech_loc = random.randint(0, self.len_wav - len_speech - 1)
        speech = np.array(
            list(struct.unpack('f' * len_speech, self.mm_wav[speech_loc * 4:speech_loc * 4 + len_speech * 4])))

        alpha = self.mix2signal(speech, noise, Snr)
        noise = alpha * noise
        mixture = noise + speech
        alpha_pow = 1 / (np.sqrt(np.sum(mixture ** 2) / len_speech + 1.0e-8) + 1.0e-8)
        speech = alpha_pow * speech
        noise = alpha_pow * noise

        sample = (Variable(torch.FloatTensor(speech.astype('float32'))),
                  Variable(torch.FloatTensor(noise.astype('float32'))),
                  )

        return sample


class TrainDataLoader(object):
    def __init__(self, s_mix_dataset, batch_size, is_shuffle=True, workers_num=16):
        self.dataloader = DataLoader(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=workers_num,
                                     collate_fn=self.collate_fn)

    def get_dataloader(self):
        return self.dataloader

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[0].size()[0], reverse=True)
        speech, noise = zip(*batch)
        speech = pad_sequence(speech, batch_first=True)
        noise = pad_sequence(noise, batch_first=True)
        mixture = speech + noise
        return [mixture, speech, noise]


class EvalDataset(Dataset):
    def __init__(self, args, filename):
        self.args = args
        self.filename = filename
        self.reader = h5py.File(self.filename, 'r')

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        reader_grp = self.reader[str(idx)]
        mixture = reader_grp['noisy_raw'][:]
        label = reader_grp['clean_raw'][:]
        sample = (Variable(torch.FloatTensor(mixture.astype('float32'))),
                  Variable(torch.FloatTensor(label.astype('float32'))),
                  )
        return sample


class EvalDataLoader(object):
    def __init__(self, s_mix_dataset, batch_size, is_shuffle=True, workers_num=16):
        self.dataloader = DataLoader(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=workers_num,
                                     collate_fn=self.collate_fn)

    def get_dataloader(self):
        return self.dataloader

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[0].size()[0], reverse=True)
        mixture, speech = zip(*batch)
        speech = pad_sequence(speech, batch_first=True)
        mixture = pad_sequence(mixture, batch_first=True)
        return [mixture, speech]
