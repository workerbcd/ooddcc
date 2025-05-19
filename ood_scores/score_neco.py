"""
The VIM score: https://arxiv.org/abs/2203.10807
Code adopted from original implementation: https://github.com/haoqiwang/vim
"""
import os, sys
import pdb
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# import faiss

from .scorer_base import ScorerBaseFeature
from utils.utils import _to_np, _to_tensor
from scipy.special import logsumexp
from tqdm import tqdm


class Neco(ScorerBaseFeature):
    def __init__(self, args):
        super().__init__(args)
        self.index = None
        self.normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

        # also need the logits of training data
        self.id_logits = np.array([])  # (N, C), the in-distribution training logits

        # params to fit
        self.u = None
        self.alpha = None

    def append_features(self, feats_new, logits_new, labels_new=None, ID=True):
        """
        append the ID features

        @param[in] feats_new         np.ndarray. A new batch of features. (N, D)
        @param[in] logits_new        np.ndarray. A new batch of logits. (N, C)
        @param[in] labels_new        np.ndarray. A new batch of GT label ids. (N, )
        """
        feats_new = _to_np(feats_new)
        logits_new = _to_np(logits_new)
        labels_new = _to_np(labels_new).astype(int)

        if ID:
            if self.id_feats.size == 0:
                self.id_feats = feats_new
                self.id_logits = logits_new
                self.id_labels = labels_new
            else:
                self.id_feats = np.concatenate(
                    (self.id_feats, feats_new),
                    axis=0
                )
                self.id_logits = np.concatenate(
                    (self.id_logits, logits_new),
                    axis=0
                )
                self.id_labels = np.concatenate(
                    (self.id_labels, labels_new),
                    axis=0
                )
            self.N, self.D = self.id_feats.shape
            self.num_class = self.id_labels.max() + 1

    def fit(self):
        if 'resnet50' in self.args.arch:
            print("cutting the dataset")
            choice_id = np.random.choice(self.id_feats.shape[0], int(0.8*self.id_feats.shape[0]))
            choice_id = np.sort(choice_id)
            self.id_feats = self.id_feats[choice_id,:]
            self.id_labels = self.id_labels[choice_id]
            self.id_logits = self.id_logits[choice_id,:]
            print("cutting done")
        self.ss = StandardScaler()
        id_feats = self.ss.fit_transform(self.id_feats)
        self.pca = PCA(self.id_feats.shape[1])
        self.pca.fit_transform(id_feats)
        self.neco_dim=self.args.neco_dim
        self.ind = 0


    def _cal_centers(self):
        centers = []
        feat = self.id_feats
        labels = self.id_labels
        for i in tqdm(range(self.num_class)):
            # fetch out the data belongs to this
            class_idx = np.where(labels == i)[0]
            class_feats = feat[class_idx[:50], :]
            # center = np.mean(class_feats,axis=0)
            centers.append(class_feats)
        return centers


    def cal_score(self, feats, logits):
        if len(feats.shape) == 1:
            feats = feats[None, :]
            logits = logits[None, :]

        max_logits = logits.max(axis=-1)
        scaled_feat = self.ss.transform(feats)
        transformed_feat = self.pca.transform(scaled_feat)
        if self.args.arch in ['deit', 'swin_b']:
            scaled_feat = feats
        reduced_feat = transformed_feat[:,:self.neco_dim]
        ss_com = np.linalg.norm(scaled_feat, axis=-1)
        sc = np.linalg.norm(reduced_feat, axis=-1)
        scores = sc/ss_com
        if 'resnet' not in self.args.arch:
            scores  = scores * max_logits

        return scores







