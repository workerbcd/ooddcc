import pdb
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
from .scorer_base import ScorerBaseFeature
from utils.utils import _to_np, _to_tensor
from utils.vis import draw_score_distributon
from scipy.special import logsumexp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap


class Density(ScorerBaseFeature):
    def __init__(self, args):
        super().__init__(args)

        # feature clipping value
        self.clip_quantile = 0.80  # 0.90   # NOTE: adopt from ViM implementation
        self.clip_max = None
        self.clip_min = None

        self.clip = None
        # print(f"max_quatile:{self.clip_quantile_max}, min_quantile:")
        # clf params
        self.w = None
        self.b = None
        self.normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)

    def VRA_clip(self, feat):
        return feat
        feat = np.where(feat < self.clip_min, np.zeros(feat.shape) + 0, feat)
        feat = np.where(feat > self.clip_max, np.zeros(feat.shape) + self.clip_max, feat)
        return feat
    def disance_map(self,feat_x,feat_y):
        if isinstance(feat_x,np.ndarray):
            nx, fx = feat_x.shape
            ny,fy = feat_y.shape
            assert fx==fy
            x2 =  (feat_x * feat_x).sum(-1)[:,None].repeat(ny,axis=1)
            y2 = (feat_y * feat_y).sum(-1)[None,:].repeat(nx,axis =0)
            xy = feat_x @ feat_y.T
            d_m = x2 +y2-2*xy
            return np.sqrt(d_m.clip(min=0))
        elif isinstance(feat_x,torch.Tensor):
            nx, fx = feat_x.size()
            ny, fy = feat_y.size()
            assert fx == fy
            x2 = (feat_x * feat_x).sum(-1)[:, None].expand(nx,ny)
            y2 = (feat_y * feat_y).sum(-1)[None, :].expand(nx,ny)
            xy = feat_x @ feat_y.T
            d_m = x2 + y2 - 2 * xy
            return torch.sqrt(d_m.clamp(min=0))
    def _cal_means(self, feats, labels):
        cls_means = []
        Nc = []
        for i in tqdm(range(self.num_class)):
            # fetch out the data belongs to this class
            class_idx = np.where(labels == i)[0]
            class_feats = feats[class_idx, :]
            Nc.append(class_feats.shape[0])
            cls_means.append(np.mean(class_feats, axis=0))
        all_mean = np.mean(feats, axis=0)
        return all_mean, cls_means, Nc

    def preprocess_train_features(self, feats):
        """Fit whitening matrix from training data & whiten the training features
        """
        print("Fiting class centers for getting Whitening projection matrix...")
        neg_direction, center_cls, _ = self._cal_means(feats, self.id_labels)
        ec = EmpiricalCovariance(assume_centered=True)
        center_samples = _to_np(center_cls)[self.id_labels, :]
        train_feats_centered = feats - center_samples
        ec.fit(train_feats_centered)
        precision = ec.precision_

        # map them
        dim =128
        eigvals, eigvecs = np.linalg.eig(precision)
        # resind = np.argsort(eigvals)
        # eigvals = eigvals[resind[dim:]]`
        # eigvecs = eigvecs[:,resind[dim:]]
        eigvals, eigvecs = self.select_direction(eigvals,eigvecs,self.normalizer(neg_direction))
        self.white_proj_mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.white_proj_mat = _to_tensor(self.white_proj_mat, device=self.device)
        feats_scaled = np.zeros_like(feats)
        print("Whitening the features...")
        for i in tqdm(range(self.N)):
            feats_scaled[i, :] = _to_np(self.white_proj_mat @ _to_tensor(feats[i, :], device=self.device))

        return feats_scaled

    def feat_distribution(self, feat, labels):
        covariances = []

        class_feature = []
        self.scalars = []
        self.pcas = []
        self.covs = []
        self.vis_feats = []
        _, feats_mean, _ = self._cal_means(feat, labels)

        for i in tqdm(range(self.num_class)):
            # fetch out the data belongs to this class
            class_idx = np.where(labels == i)[0]
            class_feats = feat[class_idx, :]
            self.vis_feats.append(class_feats[:50,:])
            # ss = StandardScaler()
            # feat_scaled = ss.fit_transform(class_feats)
            # self.scalars.append(ss)
            # ec = EmpiricalCovariance(assume_centered=False)
            # ec.fit(feat_scaled)
            # cov = ec.precision_
            # self.covs.append(cov)
            # feats_mean.append(np.mean(class_feats,axis =0))
            # pca = PCA(min(class_feats.shape))
            # pca.fit_transform(feat_scaled)
            # class_feature.append(pca.transform(np.mean(class_feats,axis=0,keepdims=True)))
            # self.pcas.appepca)
        # ec = EmpiricalCovariance(assume_centered=True)
        # feat_centerd = self.id_feats - _to_np(feats_mean)[self.id_labels,:]
        # ec.fit(feat_centerd)
        # precision = ec.precision_
        # self.covs.append(precision)
        # eigvals, eigvecs = np.linalg.eig(precision)
        # self.projs= eigvecs @ np.diag(np.power(eigvals, 0.5)) @ eigvecs.T
        # self.feat_means = _to_tensor(feats_mean, device =self.device)
        # self.projs = _to_tensor(self.projs, device=self.device)


        data_feature = np.sum(feats_mean,axis=0)


        return class_feature, data_feature

    def select_direction(self,eigval, eigvec, neg_direction):
        # print(eigval)
        map = eigvec.T @ neg_direction
        q = np.quantile(map, 0.5)
        ind = map<0
        return np.exp(-map), eigvec
        # self.projs = []
        # for cov in tqdm(self.covs):
        #     eig, eigvect = np.linalg.eig(cov)
        #     eigvect = eigvect
        #     map = eigvect @ self.data_feat
        #     q = np.quantile(map, 0.5)
        #     ind = map < 0
        #     new_eigvec = eigvect
        #     new_eig = eig
        #     proj = new_eigvec.T @ np.diag(np.power(new_eig,0.5)) @ new_eigvec
        #     self.projs.append(proj)

    def fit(self, w, b):
        # _,_,self.dist_maps, self.cls_feats = self.calss_covariaces_mean(self.id_feats,self.id_labels)
        # self.num_class=20
        id_feats = self.normalizer(self.id_feats)
        # id_feats = self.preprocess_train_features(self.id_feats)
        # _, self.data_feat = self.feat_distribution(id_feats, self.id_labels)
        _, self.feat_means, _ = self._cal_means(id_feats,self.id_labels)
        self.feat_means = _to_tensor(self.feat_means,device=self.device)
        # self.select_direction()
        # self.class_feat = _to_tensor(self.class_feat, device=self.device)
        # self.data_feat = _to_tensor(self.data_feat,device=self.device)
        # self.ss = StandardScaler()

        # self.ss.fit_transform(self.id_feats)
        # self._train_pca = PCA(self.id_feats.shape[1])
        # self._train_pca.fit_transform(self.id_feats)
        # self.avg_density = []
        # self.min_density = []
        # for m in self.dist_maps:
        #     p,_ = m.shape
        #     m = m.flatten()[:-1].view(p-1, p + 1)[:, 1:]
        #     self.avg_density.append(torch.mean(m))
        #     self.min_density.append(torch.min(m))
        # self.dist_maps = _to_np(self.dist_maps)
        self.w = w
        print(self.w.max(), self.w.min())
        self.b = b
        # self.draw()
        self.preds = []
        # self.draw_tsne()

    def draw_tsne(self):
        from sklearn.manifold import TSNE
        # import matplotlib.pyplot as plt
        tsne = TSNE(n_components=2,random_state=0)
        centers = self.vis_feats[:10]
        # centers = centers[:10]
        centers = np.concatenate(centers,axis=0)
        ps_tsne = []
        for i in range(10):
            feats = self.vis_feats[i] @ self.projs[0].T
            # print(feats.shape)
            # print(centers[i].shape)
            _ps_tsne = tsne.fit_transform(feats)
            ps_tsne.append(_ps_tsne)
        ps_tsne = np.concatenate(ps_tsne, axis=0)
        c_tsne = tsne.fit_transform(centers)
        # ps_tsne = tsne.fit_transform(cente_proj.real)
        plt.figure(figsize=(12, 10))
        plt.scatter(c_tsne[:, 0], c_tsne[:, 1], c='blue', label='full')
        # plt.scatter(ns_tsne[:, 0], ns_tsne[:, 1], c='red', label='residual')
        plt.scatter(ps_tsne[:,0], ps_tsne[:,1], c = 'green', label='proj')
        plt.legend()
        plt.savefig('figures/density_clusters.png')
        plt.close()

    def draw(self):
        path = f'figures/{self.args.arch}_weight_{self.args.id_dset}.png'
        plt.figure(figsize=(20, 10))
        sns.heatmap(self.w[:100, :], cmap='viridis')
        plt.title('Histogram of Weights')
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        plt.savefig(path)
    def draw_cls_dist(self):
        if len(self.preds)>=1:
            ood_names = [name.lower() for name in self.args.ood_dsets]
            fig_names = [self.args.id_dset] + ood_names
            path = "figures/class_dist.png"
            draw_score_distributon(self.args, self.preds, fig_names,fig_name=path)


    def _cal_logit_var(self):
        var = np.var(np.array(self.logits), axis=0)
        # self.logits = []
        print(np.mean(var))

    def post_analysis(self):
        self.draw_cls_dist()

    def cal_score(self, feats):
        # print(feats)
        # feasts = self.ss.transform(feats)
        preds = np.argmax(feats@self.w.T+self.b, axis=1)
        print(preds.shape)
        self.preds.append(preds)
        if len(feats.shape) == 1:
            feats = feats[None, :]
        dim_a, dim_b =100,170 #90, 150
        dim_data = 0   #cifar 120
        # feats = self.ss.transform(feats)
        n, f  = feats.shape
        scores = []
        # for i in range(self.num_class):
        #     feat_c = feats
        #     # feat_c = self.scalars[i].transform(feat_c)
        #     # feat_c = self.pcas[i].transform(feat_c)
        #     # feat_c = feat_c[:,dim_a:]
        #     # class_feat = self.class_feat[i][:,dim_a:]
        #     # dist = np.linalg.norm(feat_c - class_feat ,axis = 1)
        #     s = np.linalg.norm((feat_c- self.feat_means[i])@ self.projs[0].T, axis=1)
        #     scores[:, i] = s
        feats = self.normalizer(feats)
        for feat in tqdm(feats):
            feat = _to_tensor(feat,device = self.device)
            # feat = self.white_proj_mat@ feat
            d = torch.norm(feat[None,:]-self.feat_means,p=2,dim=1)
            scores.append(-d.min().cpu().numpy())
        # scores = torch.concat(scores)
        scores = _to_np(scores)
        # feats_reg =self._train_pca.transform(feats)[:, dim_data:]
        # feats_data = self._train_pca.transform(self.data_feat[None,:])[:,dim_data:]
        # reg \
        #     = np.linalg\
        #     .norm(feats_reg-feats_data,axis=1)
        scores = scores

        print(np.mean(scores))

        return scores




