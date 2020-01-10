import numpy as np
import csv
import sys
import os
import argparse
import math
import glob
import pickle
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import bayesian_layers as bl
import random
from skeleton import *
from spotting import *
torch.manual_seed(30)
np.random.seed(30)
random.seed(30)


class GestureRecognizer(nn.Module):
    def __init__(self, dropout=0.1, logstd1=-1, logstd2=-2, pi=0.5):
        super(GestureRecognizer, self).__init__()
        type = bl.ModelType.MC_DROP

        self.window = []
        self.previous = 1
        self.window_size = 3
        self.spotting = GestureSpotting()
        self.spotting.load("model_spotting3.pth")

        linear_args = {
            "mu": 0,
            "logstd1": logstd1,
            "logstd2": logstd2,
            "pi": pi,
            "type": type,
            "dropout": dropout
        }

        rnn_args = {
            "mu": 0,
            "logstd1": logstd1,
            "logstd2": logstd2,
            "pi": pi,
            "type": type,
            "dropout": dropout
        }

        last_linear_args = {
            "mu": 0,
            "logstd1": logstd1,
            "logstd2": logstd2,
            "pi": pi,
            "type": type,
            "dropout": 0
        }
        #Embedding layers
        self.bfc1 = bl.Linear(54, 32, **linear_args)
        self.bfc2 = bl.Linear(32, 16, **linear_args)
        self.fc1 = nn.Sequential(self.bfc1, nn.ReLU())
        self.fc2 = nn.Sequential(self.bfc2, nn.ReLU())

        #attributs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_state = None

        #model parameters
        self.input_size = 16
        self.output_size = 15
        self.hidden_dim = 512
        self.n_layers = 1

        #Recurrent Bayesian Layer
        self.lstm = bl.LSTM( input_size = self.input_size,\
                                    hidden_size = self.hidden_dim, num_layers = self.n_layers, \
                                    batch_first=True,**rnn_args)

        # dropout layer
        self.fc_combine = bl.Linear(self.input_size, self.input_size, **linear_args)
        self.combine = nn.Sequential(self.fc_combine, nn.ReLU())

        # Classifier layer
        self.fc = bl.Linear(self.hidden_dim, self.output_size, **last_linear_args)

    def load(self, path):
        self = super().to(self.device)
        self.fc1 = self.fc1.to(self.device)
        self.fc2 = self.fc2.to(self.device)
        self.lstm = self.lstm.to(self.device)
        self.fc = self.fc.to(self.device)

        self.load_state_dict(torch.load(path), strict=False)
        self.eval()

    def forward(self, x, hidden):

        emb = self.fc1(x)
        emb = self.fc2(emb)

        out, hidden_out = self.lstm(emb, hidden)
        out = self.fc(out)

        out = out.contiguous().view(-1, out.size(-1))
        return out, hidden_out

    def _calc_uncertainty(self, probs):
        if len(probs.shape) > 2:
            mean = probs.mean(1)
            h = -(mean * np.log(mean)).sum(1)  #entropy
        else:
            mean = probs.mean(0)
            h = -(mean * np.log(mean)).sum(0)  #entropy
        return h

    def _data_transformation(self, skl, mc_samples=20):
        data = np.expand_dims(skl, 0)
        # data = (skl-self.mean)/self.std
        #creating a batch with same data. It improves the performance of MC samples
        data = np.repeat(data, mc_samples, 0)
        data = torch.from_numpy(data).float().unsqueeze(1)
        data = data.to(self.device)
        return data

    def predict(self, skeleton):
        mc_samples = 20
        data = self._data_transformation(skeleton, mc_samples)

        spp_pred, spp_prob, spp_unc = self.spotting.predict(data)
        if len(self.window) == self.window_size: del self.window[0]
        self.window.append(spp_pred)
        w = sum(self.window) / self.window_size

        prev = self.previous

        self.previous = spp_pred
        if ((prev == 1 and w < 0.5) or (prev == 0 and w < 0.7)):
            self.hidden_state = None
            return 0, spp_prob, spp_unc

        with torch.no_grad():
            out, hidden = self.forward(data, self.hidden_state)
            out = out.cpu()
            self.hidden_state = ([
                h.data.mean(1, keepdim=True).repeat(1, mc_samples, 1) for h in hidden
            ])
            probs = F.log_softmax(out, 1).exp().detach().numpy()
            uncertainty = self._calc_uncertainty(probs)
            mean = probs.mean(0)
            pred = np.argmax(mean) + 1
            prob = mean.max()
        return pred, prob, uncertainty


if __name__ == "__main__":
    import time
    model = GestureRecognizer()
    model.load("model_gesture3_0_91.16.pth")

    # files = glob.glob("../samples/*3d.json")
    # unc = 0.0
    # total = 0.0
    # for file in files:
    #     with open(file) as f:
    #         data = json.load(f)
    #         for i, localization in enumerate(data["localizations"]):
    #             annotations = ObjectAnnotations(localization)
    #             skeletons = [Skeleton(obj) for obj in annotations.objects]
    #             for skl in skeletons:
    #                 #you can insert here a method to filter the principal skeleton
    #                 skl_normalized = skl.normalize()
    #                 skl_vector = skl_normalized.vectorized()
    #                 pred, prob, uncertainty= model.predict(skl_vector)
    #             # print(pred, prob,uncertainty)
    #                 unc += uncertainty
    #                 total+=1
    #                 #time.sleep(0.5)
    #                 break
    #         print(pred, prob, uncertainty)
    # print("mean uncertainty = ", unc/total)
