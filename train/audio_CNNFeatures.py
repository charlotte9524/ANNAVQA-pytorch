from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import os
import random
from argparse import ArgumentParser
import scipy

from scipy.io import wavfile
import torch
from scipy import signal
import numpy as np
import math

_PI = np.pi

def gga_freq_abs(x, sample_rate, freq):
  lx = len(x)
  pik_term = 2 * _PI * freq / sample_rate
  cos_pik_term = np.cos(pik_term)
  cos_pik_term2 = 2 * np.cos(pik_term)

  # number of iterations is (by one) less than the length of signal
  # Pipeline the first two iterations.
  s1 = x[0]
  s0 = x[1] + cos_pik_term2 * s1
  s2 = s1
  s1 = s0
  for ind in range(2, lx - 1):
    s0 = x[ind] + cos_pik_term2 * s1 - s2
    s2 = s1
    s1 = s0

  s0 = x[lx - 1] + cos_pik_term2 * s1 - s2

  y = np.sqrt((s0 - s1*cos_pik_term)**2 + (s1 * np.sin(pik_term))**2)
  return y


def spectrogram(x, window, window_overlap, bfs, fs):
  num_blocks = int((len(x) - window_overlap) / (len(window) - window_overlap))
  S = np.empty((len(bfs), num_blocks), dtype=np.float64)
  T = np.empty((num_blocks),dtype=np.float)

  for i in range(num_blocks):
    block = window * x[i * (len(window)-window_overlap): i * (len(window)-window_overlap) + len(window)]
    S[:, i] = gga_freq_abs(block, fs, bfs)
    T[i] = (i * (len(window)-window_overlap) + len(window)/2)/fs

  return S,T

def calcSpectrogram(audiofile):
    fs, audio = wavfile.read(audiofile)
    _, ref_audio = wavfile.read(audiofile)
    audio = (audio + 0.5)/32767.5
    audio = audio[:,0] # use left channel

    windowsize = round(fs*0.02) # 20ms
    overlap = 0.75 # 75% overlap: a 20ms window every 5ms

    window_overlap = int(windowsize*overlap)
    window = signal.get_window('hamming',windowsize,fftbins='True')

    dim = 224
    bfs = [i for i in np.arange(30,3820+(3820-30)/(dim-1),(3820-30)/(dim-1))] # bfs = 30:(3820-30)/(dim-1):3820
    bfs = np.array(bfs,dtype=float)
    bfs = 700*(pow(10,(bfs/2595))-1)
    S, t_sp= spectrogram(audio,window,window_overlap,bfs,fs)

    S = abs(np.array(S));                       # remove complex component
    S[(S==0)] = pow(2,-52);                  # no -infs in power dB
    spec_bf= np.zeros((S.shape[0],S.shape[1]))
    for i in range(len(S)):
        for j in range(len(S[i])):
            spec_bf[i][j] = math.log(S[i][j])
    return spec_bf, t_sp

class AudioDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, audios_dir, audios_names, audio_frameRate):
        super(AudioDataset, self).__init__()
        self.audios_dir = audios_dir
        self.audios_names = audios_names
        self.audios_frameRate = audio_frameRate

    def __len__(self):
        return len(self.audios_names)

    def __getitem__(self, idx):
        audios_names = self.audios_dir + self.audios_names[idx]
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        [S,T] = calcSpectrogram(audios_names)
        transforms_audio = transform(S)

        sample = {'audio_name':self.audios_names[idx], 'audio': transforms_audio, 'tStamp': T, 'frameRate': self.audios_frameRate[idx]}

        return sample


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std
        features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
        features_std = global_std_pool2d(x)
        return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(audios_data, audio_tStamp, frameRate, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()

    # 随机选取path
    patchSize = 224
    frameSkip = 2
    with torch.no_grad():
        for iFrame in range(1, int(frameRate*8),frameSkip):
            tCenter =np.argmin(abs(audio_tStamp - iFrame / frameRate))
            tStart = tCenter - patchSize / 2 + 1
            tEnd = tCenter + patchSize / 2
            if tStart < 1:
                tStart = 1
                tEnd = patchSize
            else:
                if tEnd > audios_data.shape[2]:
                    tStart = audios_data.shape[2] - patchSize + 1
                    tEnd = audios_data.shape[2]
            specRef_patch = audios_data[:, :, int(tStart-1): int(tEnd)]
            refRGB = torch.cat((specRef_patch, specRef_patch, specRef_patch),0)

            last_batch = refRGB.view(1,3,specRef_patch.shape[1],specRef_patch.shape[2]).float().to(device)
            features_mean, features_std = extractor(last_batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)

        output = torch.cat((output1, output2), 1).squeeze()
    return output

if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='LIVE_SJTU', type=str,
                        help='database name (default: LIVE_SJTU)')
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True
    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    DataBase = args.database
    if DataBase == 'LIVE_SJTU':
        ref_audios_dir = '/mnt/sdb/cyq_data/Data/LIVE-SJTU/Reference/'
        dis_audios_dir = '/mnt/sdb/cyq_data/Data/LIVE-SJTU/Video/'
        features_dir = './CNN_features_SJTU/'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    ref_audios_names, dis_audios_names = [], []
    ref_frameRate, dis_frameRate = [], []
    if DataBase == 'LIVE_SJTU':
        datainfo = '../MOS.mat'
        Info = scipy.io.loadmat(datainfo)
        ref_audios_names = [''.join(Info['refNames'][i, 1]) for i in range(0, len(Info['refNames']), 24)]
        ref_frameRate = [Info['frameRate'][i] for i in range(0, len(Info['frameRate']), 24)]
        for i in range(0,len(Info['disNames']),24):
            dis_audios_names.append(''.join(Info['disNames'][i, 1]))
            dis_audios_names.append(''.join(Info['disNames'][i+1, 1]))
            dis_audios_names.append(''.join(Info['disNames'][i+2, 1]))
            dis_frameRate.append(Info['frameRate'][i])
            dis_frameRate.append(Info['frameRate'][i+1])
            dis_frameRate.append(Info['frameRate'][i+2])

    ref_dataset = AudioDataset(ref_audios_dir, ref_audios_names, ref_frameRate)

    for i in range(len(ref_dataset)):
        current_data = ref_dataset[i]
        current_audio = current_data['audio']
        current_tStamp = current_data['tStamp']
        print('Audio {}:'.format(current_data['audio_name']))
        features = get_features(current_audio, current_tStamp, current_data['frameRate'],device)
        print(features.shape)
        np.save(features_dir + 'skip2_' + current_data['audio_name'] + '_res5', features.to('cpu').numpy())

    dis_dataset = AudioDataset(dis_audios_dir, dis_audios_names, dis_frameRate)

    for i in range(len(dis_dataset)):
        current_data = dis_dataset[i]
        current_audio = current_data['audio']
        current_tStamp = current_data['tStamp']
        print('Audio {}:'.format(current_data['audio_name']))
        features = get_features(current_audio, current_tStamp, current_data['frameRate'], device)
        print(features.shape)
        np.save(features_dir + 'skip2_' + current_data['audio_name'] + '_res5', features.to('cpu').numpy())

