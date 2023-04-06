import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo
# skvideo.setFFmpegPath(r'F:\tool\ffmpeg-N-99888-g5c7823ff1c-win64-gpl\bin')
import skvideo.io
from PIL import Image
import os
import numpy as np
import random
from argparse import ArgumentParser
import scipy
# import gc

class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, ref_videos_dir, dis_videos_dir, ref_video_names, dis_video_names, video_format='RGB', width=None, height=None, position=None, SDInfo=None):
        super(VideoDataset, self).__init__()
        self.ref_videos_dir = ref_videos_dir
        self.dis_videos_dir = dis_videos_dir
        self.ref_video_names = ref_video_names
        self.dis_video_names = dis_video_names
        self.format = video_format
        self.width = width
        self.height = height
        self.position = position
        self.SDInfo = SDInfo

    def __len__(self):
        return len(self.ref_video_names)

    def __getitem__(self, idx):
        ref_video_name = self.ref_video_names[idx]
        dis_video_name = self.dis_video_names[idx]

        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            ref_video_data = skvideo.io.vread(os.path.join(self.ref_videos_dir, ref_video_name), self.width, self.height,
                                      inputdict={'-pix_fmt': 'yuvj420p'})
            dis_video_data = skvideo.io.vread(os.path.join(self.dis_videos_dir, dis_video_name), self.width, self.height,
                                              inputdict={'-pix_fmt': 'yuvj420p'})
        else:
            ref_video_data = skvideo.io.FFmpegReader(os.path.join(self.ref_videos_dir, ref_video_name), self.height, self.width, inputdict={'-pix_fmt': 'rgb24'})
            dis_video_data = skvideo.io.FFmpegReader(os.path.join(self.dis_videos_dir, dis_video_name), self.height, self.width,
                                                 inputdict={'-pix_fmt': 'rgb24'})

        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = dis_video_data.shape[0]
        video_channel = dis_video_data.shape[3]
        video_height = dis_video_data.shape[1]
        video_width = dis_video_data.shape[2]
        transformed_dis_video = torch.zeros([video_length, video_channel, video_height, video_width])
        transformed_ref_video = torch.zeros([video_length, video_channel, video_height, video_width])
        for frame_idx in range(video_length):
            dis_frame = dis_video_data[frame_idx]
            dis_frame = Image.fromarray(dis_frame)
            dis_frame = transform(dis_frame)
            transformed_dis_video[frame_idx] = dis_frame

            ref_frame = ref_video_data[frame_idx]
            ref_frame = Image.fromarray(ref_frame)
            ref_frame = transform(ref_frame)
            transformed_ref_video[frame_idx] = ref_frame

        sample = {'ref_video_name':ref_video_name, 'dis_video_name':dis_video_name, 'dis_video': transformed_dis_video, 'ref_video':transformed_ref_video, 'sal_index':SDInfo['sal_index'][0,idx]-1}
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

def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(dis_video_data, ref_video_data, position, sal_index, frame_length, frame_interval, patchSize, patchNum, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)
    dis_output = torch.Tensor().to(device)
    ref_output = torch.Tensor().to(device)
    extractor.eval()

    ipatch = 0
    with torch.no_grad():
        for iframe in range(0, frame_length, frame_interval):
            sal_row = int(iframe/2)
            # initialize
            dis_output1 = torch.Tensor().to(device)
            dis_output2 = torch.Tensor().to(device)
            ref_output1 = torch.Tensor().to(device)
            ref_output2 = torch.Tensor().to(device)
            for idx in range(patchNum):
                patch_idx = sal_index[sal_row, idx]
                dis_batch = dis_video_data[iframe:iframe + 1, 0:3,
                        position[0][patch_idx]:position[0][patch_idx] + patchSize,
                        position[1][patch_idx]:position[1][patch_idx] + patchSize].to(device)
                ref_batch = ref_video_data[iframe:iframe + 1, 0:3,
                        position[0][patch_idx]:position[0][patch_idx] + patchSize,
                        position[1][patch_idx]:position[1][patch_idx] + patchSize].to(device)
                dis_features_mean, dis_features_std = extractor(dis_batch)
                dis_output1 = torch.cat((dis_output1, dis_features_mean), 0)
                dis_output2 = torch.cat((dis_output2, dis_features_std), 0)

                ref_features_mean, ref_features_std = extractor(ref_batch)
                ref_output1 = torch.cat((ref_output1, ref_features_mean), 0)
                ref_output2 = torch.cat((ref_output2, ref_features_std), 0)
                ipatch = ipatch + 1
                print('\r iframe: {} ipatch: {} '
                      .format(iframe, ipatch), end=' ')

            dis_output = torch.cat((dis_output, torch.cat((dis_output1.mean(axis=0, keepdim=True), dis_output2.mean(axis=0, keepdim=True)), 1)),0)
            ref_output = torch.cat(
                (ref_output, torch.cat((ref_output1.mean(axis=0, keepdim=True), ref_output2.mean(axis=0, keepdim=True)), 1)), 0)
            ipatch = 0
        dis_output = dis_output.squeeze()
        ref_output = ref_output.squeeze()
    return dis_output, ref_output

if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='LIVE_SJTU', type=str,
                        help='database name')
    parser.add_argument('--frame_length', default=192, type=int,
                        help='Total frame number')
    parser.add_argument('--frame_interval', default=2, type=int,
                        help='Frame interval for feature extraction')
    parser.add_argument('--frame_batch_size', type=int, default=5,
                        help='frame batch size for feature extraction (default: 64)')
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    database = args.database
    if database == 'LIVE_SJTU':
        ref_videos_dir = '/mnt/sdb/cyq_data/Data/LIVE-SJTU/Reference'
        dis_videos_dir = '/mnt/sdb/cyq_data/Data/LIVE-SJTU/Video'
        features_dir = './CNN_features_SJTU/'
        datainfo = '../MOS.mat'
        SDdatainfo = '../Saliency model/SJTU_position.mat'
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    dis_video_names = []
    ref_video_names = []
    SDInfo = scipy.io.loadmat(SDdatainfo)
    if database == 'LIVE_SJTU':
        Info = scipy.io.loadmat(datainfo)
        dis_video_names = [''.join(Info['disNames'][i, 0]) for i in range(0, len(Info['disNames']), 3)]
        ref_video_names = [''.join(Info['refNames'][i, 0]) for i in range(0, len(Info['refNames']), 3)]
        width = int(Info['resolution'][0][0])
        height = int(Info['resolution'][0][1])
        dis_patch = 200
        patchSize = 224
        patchNum = 25

    position_width = []
    position_height = []
    for w in range(0, width, dis_patch):
        if w < width - patchSize + 1:
            for h in range(0, height, dis_patch):
                if h < height - patchSize:
                    position_width.append(w)
                    position_height.append(h)
                else:
                    position_width.append(w)
                    position_height.append(height - patchSize)
                    break
        else:
            for h in range(0, height, dis_patch):
                if h < height - patchSize:
                    position_width.append(width - patchSize)
                    position_height.append(h)
                else:
                    position_width.append(width - patchSize)
                    position_height.append(height - patchSize)
                    break
            break

    position = [position_width, position_height]
    dis_dataset = VideoDataset(ref_videos_dir, dis_videos_dir, ref_video_names, dis_video_names, 'YUV420', width, height, position, SDInfo)

    for i in range(0, len(dis_dataset)):
        current_data = dis_dataset[i]
        current_dis_video = current_data['dis_video']
        current_ref_video = current_data['ref_video']
        sal_index = current_data['sal_index']

        dis_features, ref_features= get_features(current_dis_video, current_ref_video, position, sal_index, args.frame_length, args.frame_interval, patchSize, patchNum, device)

        ref_features = ref_features.to('cpu').numpy()
        dis_features = dis_features.to('cpu').numpy()
        features = np.concatenate((ref_features, dis_features), 0)
        print('Distorted Video {}'.format(current_data['dis_video_name']))
        print('Referenced Video {}'.format(current_data['ref_video_name']))
        print('Feature length {}'.format(features.shape[0]))
        np.save(features_dir + 'skip2_SD_' + current_data['dis_video_name'] + '_res5', features)

