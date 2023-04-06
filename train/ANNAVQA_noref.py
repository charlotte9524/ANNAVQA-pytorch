from argparse import ArgumentParser
import os
import scipy.io
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats
from tensorboardX import SummaryWriter
import datetime
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

class VQADataset(Dataset):
    def __init__(self, features_dir, allindex, dis_names, allmos, seg_num=4, seg_len=24, feat_dim=4096):
        super(VQADataset, self).__init__()
        self.allindex = allindex
        self.dis_names = dis_names
        self.seg_len = seg_len
        self.seg_num = seg_num
        self.features_dir = features_dir
        self.feat_dim = feat_dim
        self.allmos = allmos

    def __len__(self):
        return len(self.allindex)

    def __getitem__(self, index):
        features = np.zeros((self.seg_num, self.seg_len, self.feat_dim))
        afeatures = np.zeros((self.seg_num, self.seg_len, self.feat_dim))
        tmpvideo = np.load(self.features_dir + 'skip2_SD_' + self.dis_names[self.allindex[index]][0] + '_res5.npy')
        tmpaudio = np.load(self.features_dir + 'skip2_' + self.dis_names[self.allindex[index]][1] + '_res5.npy')
        tmp_length = int(len(tmpvideo) / 2)
        for j in range(seg_num):
            features[j, :, :] = tmpvideo[tmp_length+j * self.seg_len:tmp_length+(j + 1) * self.seg_len, :self.feat_dim]
            afeatures[j, :, :] = tmpaudio[j * self.seg_len:(j + 1) * self.seg_len, :self.feat_dim]
        sample = features, afeatures, self.seg_len, np.array([self.allmos[self.allindex[index]]])
        return sample

class ANN(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)  #
        self.dropout = nn.AlphaDropout(p=dropout_p)  #
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers - 1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input

class VSFA(nn.Module):
    def __init__(self, input_size=4096, min_len=4839, reduced_size=2048, hidden_size=1024):
        super(VSFA, self).__init__()
        self.hidden_size = hidden_size
        self.min_len = min_len
        self.video_ann = ANN(input_size, reduced_size, 1)
        self.video_rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.video_q1 = nn.Linear(hidden_size, 512)
        self.video_relu = nn.ReLU()
        self.video_dro = nn.Dropout()
        self.video_q2 = nn.Linear(512, 1)

        self.audio_ann = ANN(input_size, reduced_size, 1)
        self.audio_rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.audio_q1 = nn.Linear(hidden_size, 512)
        self.audio_relu = nn.ReLU()
        self.audio_dro = nn.Dropout()
        self.audio_q2 = nn.Linear(512, 1)

        self.fc1 = nn.Linear(min_len, 32)  #min_len
        self.relu1 = nn.ReLU()
        self.dro1 = nn.Dropout()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dro2 = nn.Dropout()
        self.fc3 = nn.Linear(16, 1)

    def forward(self, features, afeatures):
        video_input = self.video_ann(features)  # dimension reduction
        audio_input = self.audio_ann(afeatures)
        video_outputs, _ = self.video_rnn(video_input, self._get_initial_state(video_input.size(0), video_input.device))
        audio_outputs, _ = self.audio_rnn(audio_input, self._get_initial_state(audio_input.size(0), audio_input.device))
        video_q1 = self.video_q1(video_outputs)
        audio_q1 = self.audio_q1(audio_outputs)
        video_relu = self.video_relu(video_q1)
        audio_relu = self.audio_relu(audio_q1)
        video_dro = self.video_dro(video_relu)
        audio_dro = self.audio_dro(audio_relu)
        video_q = self.video_q2(video_dro)
        audio_q = self.audio_q2(audio_dro)
        fc1 = self.fc1(torch.cat([video_q.squeeze(dim=2), audio_q.squeeze(dim=2)], dim=1))
        relu1 = self.relu1(fc1)
        dro1 = self.dro1(relu1)
        fc2 = self.fc2(dro1)
        relu2 = self.relu2(fc2)
        dro2 = self.dro2(relu2)
        score = self.fc3(dro2)
        return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


def logistic_func(X, beta1, beta2, beta3, beta4, beta5):
    logisticPart = 1 + np.exp(np.multiply(beta2,X - beta3))
    yhat = beta1* (0.5-np.divide(1,logisticPart)) + np.multiply(beta4,X) + beta5
    return yhat

def compute_metrics(y_pred, y):
    '''
    compute metrics btw predictions & labels
    '''
    # compute SRCC & KRCC
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

    # logistic regression btw y_pred & y
    beta_init = [10, 0, np.mean(y_pred), 0.1, 0.1]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))  #
    y_pred_logistic = logistic_func(y_pred, *popt)

    # compute  PLCC RMSE
    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
    return [SRCC, KRCC, PLCC, RMSE]

if __name__ == "__main__":
    parser = ArgumentParser(description='ANNAVQA NR')
    parser.add_argument("--seed", type=int, default=19990524)
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 2000)')
    parser.add_argument('--database', default='LIVE-SJTU', type=str,
                        help='database name (default: LIVE-SJTU)')
    parser.add_argument('--exp_id', default='ANNAVQA_noref', type=str,
                        help='exp id (default: ANNAVQA_noref)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--TestEpoch', default=10, type=int,
                        help='Epoch for test (default: 10)')
    parser.add_argument('--TestRatio', default=0.2, type=float,
                        help='test ratio of dataset (default: 0.2)')
    parser.add_argument('--ValidRatio', default=0.2, type=float,
                        help='test ratio of dataset (default: 0.2)')
    parser.add_argument("--notest_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    print('EXP ID: {}'.format(args.exp_id))

    features_dir = './CNN_features_SJTU/'
    model_dir = './models/'
    datainfo = '../MOS.mat'
    seq_num = 24  # The number of distorted A/V sequences produced by the same referenced A/V sequence
    features_len = 96 # video/audio features length
    feat_dim = 4096 # video/audio feature dim
    seg_num = 4 # Short seq no. in the SMAM

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = scipy.io.loadmat(datainfo)
    disNames = []
    for i in range(0, len(Info['disNames'])):
        disNames.append([''.join(Info['disNames'][i, 0]), ''.join(Info['disNames'][i, 1]), Info['frameNum'][i, 0]])

    MOSz = []
    for i in range(len(Info['MOS'])):
        MOSz.append(float(Info['MOS'][i][0] / 100))

    bs = args.batch_size
    test_epoch = args.TestEpoch
    test_ratio = args.TestRatio
    valid_ratio = args.ValidRatio
    exp_id = args.exp_id
    seg_len = int(features_len / seg_num)
    TGNum = int(len(disNames) / seq_num * test_ratio)
    VGNum = int(len(disNames) / seq_num * valid_ratio)

    train_time =  datetime.datetime.now().strftime("%I%M%p_%B%d%Y")
    model_dir = '{}EXP{}-{}'.format(model_dir, exp_id, train_time)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not args.disable_visualization:  # Tensorboard Visualization
        writer = SummaryWriter(log_dir='{}/EXP{}-{}-{}-{}-{}-{}-seg{}'
                               .format(args.log_dir, exp_id, args.database, args.lr, args.batch_size, args.epochs, train_time, seg_num))
    save_result_file = 'results/EXP{}-{}'.format(exp_id, train_time)
    if not os.path.exists('results'):
        os.makedirs('results')
    all_result = np.zeros((test_epoch, 4))
    args_epochs = args.epochs

    for te in range(test_epoch):
        trained_model_file = 'models/EXP{}-{}-{}-{}-{}-{}-seg{}-epoch{}' \
            .format(args.exp_id, args.database, args.lr, args.batch_size, args.epochs, train_time, seg_num, te,)
        allindex = [i for i in range(len(disNames))]
        model = VSFA(input_size=feat_dim, min_len=2*seg_len).to(device)  #

        group_name = str(te)
        test_num = []
        test_index = []
        valid_num = []
        valid_index = []
        train_index = allindex
        while (len(valid_num) < VGNum):
            x = random.randint(0, len(disNames) / seq_num - 1)
            if x not in valid_num:
                group_name = group_name + '_Valid:' + str(x)
                valid_num.append(x)
                for ind in range(x * seq_num, x * seq_num + seq_num):
                    valid_index.append(ind)
                    train_index.remove(ind)

        while(len(test_num)<TGNum):
            x = random.randint(0, len(disNames) / seq_num - 1)
            if x not in test_num and x not in valid_num:
                group_name = group_name + '_Test:' + str(x)
                test_num.append(x)
                for ind in range(x * seq_num, x * seq_num + seq_num):
                    test_index.append(ind)
                    train_index.remove(ind)

        train_dataset = VQADataset(features_dir, train_index, disNames, MOSz, seg_num, seg_len, feat_dim)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   pin_memory=False, num_workers=0)
        valid_dataset = VQADataset(features_dir, valid_index, disNames, MOSz, seg_num, seg_len, feat_dim)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False,
                                                   pin_memory=False, num_workers=0)
        test_dataset = VQADataset(features_dir, test_index, disNames, MOSz, seg_num, seg_len, feat_dim)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                                   pin_memory=False, num_workers=0)

        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        prev_RMSE = 100
        epoch = 0

        while (epoch < args_epochs):
            # Train
            model.train()
            L = 0
            for j in range(seg_num):
                for i, (features, afeatures, length, label) in enumerate(train_loader):
                    print("\r train_index:{}".format(i), end='')
                    features = features[:, j, :, :]
                    afeatures = afeatures[:, j, :, :]

                    features = features.to(device).float()
                    afeatures = afeatures.to(device).float()
                    label = label.to(device).float()
                    optimizer.zero_grad()
                    outputs = model(features, afeatures)
                    loss = criterion(outputs, label)
                    loss.backward()
                    optimizer.step()

                    L = L + loss.item()
                train_loss = L / (i + 1)

            if not args.disable_visualization:  # record training curves
                writer.add_scalar("loss/train-{}".format(group_name), train_loss, epoch)  #

            if epoch >= 30:
                model.eval()
                # val
                y_pred = np.zeros(len(valid_loader))
                y_val = np.zeros(len(valid_loader))
                L = 0
                with torch.no_grad():
                    for i, (tmpfeatures, tmpafeatures, length, label) in enumerate(valid_loader):
                        print("\r valid_index:{}".format(i), end='')
                        y_val[i] = label.item()
                        for j in range(seg_num):
                            features = tmpfeatures[:, j, :, :]
                            afeatures = tmpafeatures[:, j, :, :]
                            features = features.squeeze(dim=1)
                            afeatures = afeatures.squeeze(dim=1)

                            features = features.to(device).float()
                            afeatures = afeatures.to(device).float()
                            label = label.to(device).float()
                            outputs = model(features, afeatures)
                            y_pred[i] = y_pred[i] + outputs.item()

                        y_pred[i] = y_pred[i] / seg_num

                    [val_SROCC, val_KROCC, val_PLCC, val_RMSE] = compute_metrics(y_pred, y_val)
                    print("Val results: {}-{}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                          .format(te, epoch, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
                if not args.disable_visualization:
                    writer.add_scalar("SROCC/valid-{}".format(group_name), val_SROCC, epoch)  #
                    writer.add_scalar("KROCC/valid-{}".format(group_name), val_KROCC, epoch)  #
                    writer.add_scalar("PLCC/valid-{}".format(group_name), val_PLCC, epoch)  #
                    writer.add_scalar("RMSE/valid-{}".format(group_name), val_RMSE, epoch)  #

                if val_RMSE < prev_RMSE:
                    prev_RMSE = val_RMSE
                    torch.save(model.state_dict(), trained_model_file)

                    if not args.notest_during_training:
                        y_pred = np.zeros(len(test_loader))
                        y_test = np.zeros(len(test_loader))
                        L = 0
                        with torch.no_grad():
                            for i, (tmpfeatures, tmpafeatures, length, label) in enumerate(test_loader):
                                print("\r test_index:{}".format(i), end='')
                                y_test[i] = label.item()  #
                                for j in range(seg_num):
                                    features = tmpfeatures[:, j, :, :]
                                    afeatures = tmpafeatures[:, j, :, :]
                                    features = features.squeeze(dim=1)
                                    afeatures = afeatures.squeeze(dim=1)

                                    features = features.to(device).float()
                                    afeatures = afeatures.to(device).float()
                                    label = label.to(device).float()
                                    outputs = model(features, afeatures)
                                    y_pred[i] = y_pred[i] + outputs.item()

                                y_pred[i] = y_pred[i] / seg_num

                        [test_SROCC, test_KROCC, test_PLCC, test_RMSE] = compute_metrics(y_pred, y_test)

                        print("EXP ID={}: Update best model using best_val_criterion in epoch {}"
                              .format(args.exp_id, epoch))
                        print("Tes results: {}-{}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                              .format(te, epoch, test_SROCC, test_KROCC, test_PLCC, test_RMSE))
                        if not args.disable_visualization:
                            writer.add_scalar("SROCC/test-{}".format(group_name), test_SROCC, epoch)  #
                            writer.add_scalar("KROCC/test-{}".format(group_name), test_KROCC, epoch)  #
                            writer.add_scalar("PLCC/test-{}".format(group_name), test_PLCC, epoch)  #
                            writer.add_scalar("RMSE/test-{}".format(group_name), test_RMSE, epoch)  #

            epoch = epoch + 1
        if not args.notest_during_training:
            all_result[te] = [test_SROCC, test_KROCC, test_PLCC, test_RMSE]
    print(all_result)
    if not args.notest_during_training:
        np.save(save_result_file, all_result)
        print('Average result: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}'
              .format(np.mean(all_result[:,0]),np.mean(all_result[:,1]),np.mean(all_result[:,2]),np.mean(all_result[:,3])))