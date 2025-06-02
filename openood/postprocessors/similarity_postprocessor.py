from typing import Any
import torch
from .base_postprocessor import BasePostprocessor

class SimilarityPostprocessor(BasePostprocessor):
    """
    OOD postprocessor using representational similarity under input perturbation.
    Computes metrics based on activation consistency across noisy samples.
    """
    def __init__(self, config):
        super().__init__(config)
        args = config.postprocessor.postprocessor_args
        self.num_samples = getattr(args, 'num_samples', 100)
        self.noise_magnitude = getattr(args, 'noise_magnitude', 0.01)

    @torch.no_grad()
    def postprocess(self, net, data):
        # collect features for clean and noisy inputs
        output, feat0 = net(data, return_feature=True)
        feat_list = [feat0]
        for _ in range(self.num_samples - 1):
            noise = torch.normal(mean=torch.zeros_like(data), std=self.noise_magnitude)
            _, feat_i = net(data + noise, return_feature=True)
            feat_list.append(feat_i)
        # stack features: (N, batch_size, D)
        feat_stack = torch.stack(feat_list, dim=0)
        N, batch_size, D = feat_stack.shape
        eps = 1e-12
        metrics = []
        # 1. average pairwise cosine similarity
        cosines = []
        for j in range(batch_size):
            f = feat_stack[:, j, :]
            f_norm = f / (f.norm(dim=1, keepdim=True) + eps)
            sim_mat = f_norm @ f_norm.T
            avg_sim = (sim_mat.sum() - N) / (N * (N - 1))
            cosines.append(avg_sim)
        metrics.append(torch.stack(cosines))
        # 2. Cronbach's alpha
        alphas = []
        for j in range(batch_size):
            f = feat_stack[:, j, :]
            var_dims = f.var(dim=0, unbiased=False)
            total_scores = f.sum(dim=1)
            var_total = total_scores.var(dim=0, unbiased=False)
            alpha = (D / (D - 1)) * (1 - var_dims.sum() / (var_total + eps))
            alphas.append(alpha)
        metrics.append(torch.stack(alphas))
        # 3. average Jaccard similarity on binary masks
        jaccards = []
        for j in range(batch_size):
            f = feat_stack[:, j, :]
            mask = (f > 0).float()
            inter = mask @ mask.T
            union = mask.sum(dim=1, keepdim=True) + mask.sum(dim=1, keepdim=True).T - inter
            jaccard_mat = inter / (union + eps)
            avg_j = (jaccard_mat.sum() - N) / (N * (N - 1))
            jaccards.append(avg_j)
        metrics.append(torch.stack(jaccards))
        # 4. CKA via top singular value fraction
        ckas = []
        for j in range(batch_size):
            f = feat_stack[:, j, :]
            G = f @ f.T
            S = torch.linalg.svdvals(G)
            frac = S[0] / (S.sum() + eps)
            ckas.append(frac)
        metrics.append(torch.stack(ckas))
        # 5. negative MMD (squared difference between half means)
        mmds = []
        half = N // 2
        for j in range(batch_size):
            f = feat_stack[:, j, :]
            mean1 = f[:half].mean(dim=0)
            mean2 = f[half:].mean(dim=0)
            mmd_val = ((mean1 - mean2) ** 2).sum()
            mmds.append(-mmd_val)
        metrics.append(torch.stack(mmds))
        # build metric labels
        labels = [
            'avg_cosine_similarity',
            'cronbach_alpha',
            'avg_jaccard_similarity',
            'cka_rank1_fraction',
            'neg_mmd'
        ]
        self.metric_labels = labels
        preds = output.argmax(dim=1)
        return preds, metrics 