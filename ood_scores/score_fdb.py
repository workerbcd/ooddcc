import pdb
import numpy as np
import torch
from tqdm import tqdm

from .scorer_base import ScorerBaseFeature
from utils.utils import _to_np, _to_tensor
from scipy.special import logsumexp
import torch.nn.functional as F


class FDB(ScorerBaseFeature):
    def __init__(self, args):
        super().__init__(args)

        # feature clipping value
        self.clip_quantile = 0.90  # 0.90   # NOTE: adopt from ViM implementation
        self.args = args
        self.clip = None

        # self.p = 10
        self.weight_p = 10
        if args.id_dset =='imagenet':
            self.clip_threshold = 0.8
        elif 'CIFAR' in args.id_dset:
            self.clip_threshold=1.0
        self.masked_w = None
        self.mask_f = None

        self.w = None
        self.b = None
        if args.id_dset=='imagenet':
            self.p = 10
            self.weight_p =10
        elif args.id_dset=='CIFAR-10':
            self.p = 90
            self.weight_p=90
        elif args.id_dset=='CIFAR-100':
            self.p=10
            self.weight_p=90
        print(self.p)
    def _cal_center(self):
        load_file = f'./featCache/{self.args.id_dset}_{self.args.arch}_taylor_mean_class.npy'
        self.info = np.load(load_file) # size: 2048
    def calculate_shap_value(self, w,b):
        self.contrib = self.info.T
        self.weight = _to_tensor(w,device=self.device)
        self.out_features, self.in_features = self.weight.size()
        self.mask_f = torch.zeros(self.out_features, self.in_features)
        self.masked_w = torch.zeros((self.out_features, self.out_features, self.in_features))

        for class_num in range(self.out_features):
            self.matrix = abs(self.contrib[class_num, :]) * self.weight.data.cpu().numpy()
            self.thresh = np.percentile(self.matrix, self.weight_p)
            mask_w = torch.Tensor((self.matrix > self.thresh))
            self.masked_w[class_num, :, :] = (self.weight.squeeze().cpu() * mask_w).cuda()
            self.class_thresh = np.percentile(self.contrib[class_num, :], self.p)
            self.mask_f[class_num, :] = torch.Tensor((self.contrib[class_num, :] > self.class_thresh))
        self.bias = _to_tensor(b,device=self.device)
    def _forward_line(self,input):
        input = input[None,:]
        pre = input[:, None, :] * self.weight
        if self.bias is not None:
            pred = pre.sum(2) + self.bias
        else:
            pred = pre.sum(2)
        pred = torch.nn.functional.softmax(pred, dim=1)
        preds = np.argmax(pred.cpu().detach().numpy(), axis=1)

        counter_cp = 0
        cp = torch.zeros((len(input), self.in_features)).cuda()
        for idx in preds:
            cp[counter_cp, :] = input[counter_cp, :] * self.mask_f[idx, :].cuda()
            counter_cp = counter_cp + 1

        vote = torch.zeros((len(preds), self.out_features, self.in_features)).cuda()
        counter_dice = 0
        for idx in preds:
            vote[counter_dice, :, :] = cp[counter_dice, :] * self.masked_w[idx, :, :].cuda()
            counter_dice = counter_dice + 1

        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out
    def fit(self, w, b):
        self.train_mean = np.mean(self.id_feats, axis=0)

        denominator_matrix = np.zeros((self.num_class, self.num_class))#.to(device)
        for p in tqdm(range(self.num_class)):
            w_p = w - w[p, :]
            denominator = np.linalg.norm(w_p, axis=1)
            denominator[p] = 1
            denominator_matrix[p, :] = denominator
        self.d_mat = denominator_matrix
        self.w = w
        self.b =b
        # self.w = np.clip(w, a_min=clip_w_min, a_max=clip_w)
        # self.b = np.clip(b, a_min=clip_b_min, a_max=clip_b)
    def _cal_logit_var(self):
        var = np.var(np.array(self.logits), axis=0)
        self.logits = []
        print(np.mean(var))
    def cal_score(self, feats):
        # print(feats.shape)
        if len(feats.shape) == 1:
            feats = feats[None, :]
        logits = feats@ self.w.T +self.b
        values = np.max(logits,axis=1)
        nnid = np.argmax(logits,axis=1)
        logits_sub = np.abs(logits - values[:,None])
        scores = np.sum(logits_sub/self.d_mat[nnid],axis=1)
        reg = np.linalg.norm(feats - self.train_mean,axis =1)
        return scores/reg