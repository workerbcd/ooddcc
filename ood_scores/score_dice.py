import pdb
import numpy as np

from .scorer_base import ScorerBaseFeature
from utils.utils import _to_np, _to_tensor
from scipy.special import logsumexp
import seaborn as sns
import matplotlib.pyplot as plt
import os


class DICE(ScorerBaseFeature):
    def __init__(self, args):
        super().__init__(args)

        # feature clipping value
        self.clip_quantile = 0.90  # 0.90   # NOTE: adopt from ViM implementation
        self.args = args
        self.clip = None

        # clf params
        # if args.arch == 'resnet50':
        #     self.info = np.load(f"featCache/{args.id_dset}_{args.arch}_feat_stat.npy")
        #     print(self.info.shape)
        # elif args.arch == 'vit_b':
        #     self.info = np.array([1.0]*768)
        self.w = None
        self.b = None
        self.clip_max=None
        if args.id_dset=='imagenet':
            self.clip_max = 1.0
            self.clip_min = 0.5
            self.p = 70
        else:
            self.clip_max_quantile=0.95
            self.clip_min_quantile=0.6
            self.p = 90
        print(self.p)
    def _cal_center(self,w):
        load_file = f'./featCache/{self.args.id_dset}_{self.args.arch}_feat_stat.npy'
        if os.path.exists(load_file):
            self.info = np.load(load_file) # size: 2048
        else:
            # weight_w = self.train_mean*w
            self.info =self.train_mean
    def VRA_clip(self,feat):
        feat = np.where(feat<self.clip_min,np.zeros(feat.shape)+0, feat)
        feat = np.where(feat>self.clip_max, np.zeros(feat.shape)+self.clip_max, feat)
        return feat
    def fit(self, w, b):
        self.train_mean = np.mean(self.id_feats, axis=0)
        self._cal_center(w)
        # self.clip = np.quantile(self.id_feats, self.clip_quantile)
        if self.clip_max==None:
            self.clip_max = np.quantile(self.id_feats,self.clip_max_quantile,axis=0)
            self.clip_min = np.quantile(self.id_feats,self.clip_min_quantile,axis=0)
        self.contrib =self.info[None,:]*w
        self.thre = np.percentile(self.contrib, self.p)
        mask = np.array((self.contrib>self.thre))
        meanfeat = self.train_mean @ w.T +b
        self.w = w*mask
        mean_feat_ = self.train_mean @ self.w.T +b
        delt_b = mean_feat_ - meanfeat
        print(f'max_value: {np.max(self.w)}, min_value: {np.min(self.w)}')
        print(delt_b)
        self.b = b #+ delt_b
        self.logits=[]
        # self.draw()
        # self.w = np.clip(w, a_min=clip_w_min, a_max=clip_w)
        # self.b = np.clip(b, a_min=clip_b_min, a_max=clip_b)
    def _cal_logit_var(self):
        var = np.var(np.array(self.logits), axis=0)
        self.logits = []

        print(np.mean(var))
    def draw(self):
        path = f'figures/dice_weight_{self.args.id_dset}.png'
        plt.figure(figsize=(20,10))
        sns.heatmap(self.w[:100,:], cmap='viridis')
        plt.title('Histogram of Weights')
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        plt.savefig(path)

    def cal_score(self, feats):
        print(feats.shape)
        if len(feats.shape) == 1:
            feats = feats[None, :]
        # logit_clip = np.clip(feats, a_min=None, a_max=self.clip) @ self.w.T + self.b
        # feats = self.VRA_clip(feats)
        logit = feats @ self.w.T + self.b
        self.logits=logit
        scores = logsumexp(logit, axis=1)
        # scores = logsumexp(logit, axis=-1)
        # scores = np.max(logit_clip, axis=-1)
        # logit = feats  @ self.w.T + self.b
        # scores = np.sum(np.abs(logit - np.max(logit, axis=1)[:, None]), axis=-1)
        reg = np.linalg.norm(feats - self.info, axis=1)
        return scores