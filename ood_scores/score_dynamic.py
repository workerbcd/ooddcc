import pdb
import numpy as np
import torch

from .scorer_base import ScorerBaseFeature
from utils.utils import _to_np, _to_tensor

from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance


class Dynamic(ScorerBaseFeature):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.normalizer = lambda x: x/torch.norm(x,p=2, dim=-1,keepdim=True) if \
            isinstance(x, torch.Tensor) else x / np.linalg.norm(x, axis=-1, keepdims=True)
        self.count = 0


    def _cal_means(self, feats, labels):
        _, f = feats.shape
        cls_means = np.zeros((self.num_class, f))
        Nc = []
        for i in tqdm(range(self.num_class)):
            class_idx = np.where(labels == i)[0]
            class_feats = feats[class_idx, :]
            cls_mean = np.mean(class_feats, axis=0)
            cls_means[i] = cls_mean

        all_mean = np.mean(feats, axis=0)
        cls_means = _to_tensor(cls_means, device=self.device)
        return all_mean, cls_means, Nc


    def collect_center_sample(self, feats, labels):
        all_mean, center_cls, _ = self._cal_means(feats, labels)
        center_samples = _to_np(center_cls)[self.id_labels, :]
        return center_samples, center_cls, all_mean
    def _cal_precision_(self, feats):
        print("Fiting prior matrix")
        estimator = EmpiricalCovariance(assume_centered=True)
        self.data_num, f = feats.shape
        estimator.fit(feats)
        cov = estimator.covariance_
        eigval,eigvec = np.linalg.eig(cov)
        sort_id = np.argsort(eigval)
        arch = self.args.arch.lower()
        if arch == 'densenet':
            dim= 300
        elif arch == 'wrn':
            dim=50
        elif self.args.id_dset == 'imagenet':
            dim=512
        else:
            print(f"{arch} is not implemented")
            raise NotImplementedError

        eigval = eigval[sort_id[:dim]]
        eigvec = eigvec[:, sort_id[:dim]]

        precision = np.linalg.inv(cov)
        self.basis = _to_tensor(eigvec.T, device=self.device)
        self.precison = _to_tensor(precision,device=self.device)
        self.cov = _to_tensor(cov,device=self.device)
    def fit(self):
        self.id_feats = self.normalizer(self.id_feats)
        center_samples, self.cls_means, _ = self.collect_center_sample(self.id_feats, self.id_labels)
        r_feat = self.id_feats - center_samples
        self._cal_precision_(r_feat)

    def cal_score(self, feats):
        scores = []
        feats = self.normalizer(feats)
        # Idmatrix = _to_tensor(np.eye(f), device=self.device)
        for feat in tqdm(feats):
            feat = _to_tensor(feat, device=self.device)#[None,:]
            feat = self.normalizer(feat)
            a = torch.einsum('i, bi->b', feat, self.basis)[None,:]
            tcov = self.basis.T @ a.T @ a @ self.basis
            pm = torch.linalg.inv((self.cov-tcov))
            d = feat- self.cls_means
            dists = torch.einsum('bi,ij,bj->b',d,pm, d)
            d_min = torch.min(torch.sqrt(dists))
            score = -d_min
            scores.append(score.cpu().numpy())
        scores = _to_np(scores)
        print(np.mean(scores), np.var(scores))
        return scores
