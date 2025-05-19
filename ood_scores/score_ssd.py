import pdb
import numpy as np
import torch

from .scorer_base import ScorerBaseFeature
from utils.utils import _to_np, _to_tensor

from tqdm import tqdm




def percentile(t: torch.tensor, q: float) :
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result

class SSD(ScorerBaseFeature):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self.normalizer = lambda x: x/torch.norm(x,p=2, dim=-1,keepdim=True) if \
            isinstance(x, torch.Tensor) else x / np.linalg.norm(x, axis=-1, keepdims=True)
        self.count = 0

    def _cal_means(self, feats, labels):
        _, f = feats.shape
        self.cls_means = []
        Nc = []
        self.invs = []
        for i in tqdm(range(self.num_class)):
            class_idx = np.where(labels == i)[0]
            class_feats = feats[class_idx, :]
            cls_mean = np.mean(class_feats, axis=0,keepdims=True)
            self.cls_means.append(cls_mean)
            inv = np.linalg.pinv(np.cov(class_feats.T, bias=True))
            self.invs.append(inv)


    def fit(self):
        self.id_feats = self.normalizer(self.id_feats)

        self.mean, self.std = np.mean(self.id_feats, axis=0, keepdims=True), np.std(self.id_feats,axis=0,keepdims=True)
        self.id_feats = (self.id_feats - self.mean)/self.std


        self._cal_means(self.id_feats, self.id_labels)




    def cal_score(self, feats):
        scores = []
        feats = self.normalizer(feats)

        for i in tqdm(range(self.num_class)):
            score = np.sum(
            (feats - self.cls_means[i])
            * (
                self.invs[i].dot(
                    (feats - self.cls_means[i]).T
                )
            ).T,
            axis=-1,
        )
            scores.append(score)
        scores = _to_np(scores)
        scores = np.min(scores,axis=0)
        print(np.mean(scores), np.var(scores), scores.shape)
        return scores
