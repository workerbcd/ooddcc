"""
The ReAct+Energy score: https://arxiv.org/abs/2111.12797
Code adopted from: https://github.com/haoqiwang/vim 
"""
import pdb
import numpy as np

from .scorer_base import ScorerBaseFeature 
from utils.utils import _to_np, _to_tensor
from scipy.special import logsumexp

class ReAct(ScorerBaseFeature):
    def __init__(self, args):
        super().__init__(args)

        # feature clipping value
        self.clip_quantile = 0.99#0.90   # NOTE: adopt from ViM implementation
        self.clip_quantile_min = 0.05
        self.clip = None

        # clf params
        self.w = None
        self.b = None

    def fit(self, w, b):
        self.clip = np.quantile(self.id_feats, self.clip_quantile)
        self.w = w
        self.b = b
    def fit_(self, w,b):
        # self.clip = np.quantile(self.id_feats, self.clip_quantile)
        clip_w =np.quantile(w, self.clip_quantile)
        clip_w_min = np.quantile(w,self.clip_quantile_min)
        clip_b = np.quantile(b,self.clip_quantile)
        clip_b_min = np.quantile(b,self.clip_quantile_min)
        print("min:", self.clip_quantile_min)
        print("max:", self.clip_quantile)

        self.w = np.clip(w,a_min=clip_w_min,a_max=clip_w)
        self.b = np.clip(b,a_min=clip_b_min, a_max=clip_b)
        
        
    def cal_score(self, feats):
        if len(feats.shape) == 1:
            feats = feats[None, :]
        logit_clip = np.clip(feats, a_min=None, a_max=self.clip) @ self.w.T + self.b
        # logit_clip = feats @ self.w.T + self.b
        scores = logsumexp(logit_clip, axis=-1)
        # scores = np.max(logit_clip, axis=-1)
        return scores