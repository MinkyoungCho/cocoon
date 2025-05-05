import torch
import torch.nn as nn


class CrossModalCenterLoss(nn.Module):
    """Center loss.
    This class is basically derived from https://github.com/LongLong-Jing/Cross-Modal-Center-Loss/tree/main.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=128, use_gpu=True, mode=None):
        super(CrossModalCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers0 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            self.centers1 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            self.centers2 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            self.centers3 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            self.centers4 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            self.centers5 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers0 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            self.centers1 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            self.centers2 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            self.centers3 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            self.centers4 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            self.centers5 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

        assert mode in ["train", "test"]

    def forward(self, labels, img_feats, pts_feats, dec_layer_idx=None):
        """
        Args:
            X_feats: feature matrix with shape (num_queries, feat_dim).
            labels: ground truth labels with shape (num_queries).
        """

        num_queries = img_feats.size(1)

        # img_feats: [900, 128] --> [33, 128]
        # pts_feats: [900, 128] --> [33, 128]
        # selc.centers: [10, 128] --> [33, 128]
        img_feats = img_feats.squeeze()
        pts_feats = pts_feats.squeeze()
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(num_queries, self.num_classes)
        mask = labels.eq(classes.unsqueeze(0))
        
        self_centers = getattr(self, f'centers{dec_layer_idx}')

        centers = self_centers.unsqueeze(0).expand(num_queries, -1, -1)
        centers_masked = centers[mask]  # [33, 128]

        img_feats_masked = img_feats[mask.sum(dim=1) == 1]  # [33, 128]
        pts_feats_masked = pts_feats[mask.sum(dim=1) == 1]  # [33, 128]

        # assert we have correct feature-center pairs
        indices = torch.nonzero(mask.sum(dim=1) == 1).squeeze(0)

        if len(indices) > 1:
            assert img_feats_masked.shape[0] == len(indices)
            assert pts_feats_masked.shape[0] == len(indices)
            assert centers_masked.shape[0] == len(indices)
            assert (centers[0][labels[indices][0][0]] == centers_masked[0]).all().item()
            assert (centers[0][labels[indices][1][0]] == centers_masked[1]).all().item()
            assert (img_feats[indices[0]] == img_feats_masked[0]).all().item()
            assert (img_feats[indices[1]] == img_feats_masked[1]).all().item()
            assert (pts_feats[indices[0]] == pts_feats_masked[0]).all().item()
            assert (pts_feats[indices[1]] == pts_feats_masked[1]).all().item()

        loss_center = self._calculate_center_loss(
            img_feats_masked, pts_feats_masked, centers_masked, labels, num_queries
        )

        # loss_geomed
        loss_geomed = self._calculate_geomed_loss(
            img_feats_masked, pts_feats_masked, centers_masked, labels, num_queries
        )
        
        loss_sep = self._calculate_separate_loss(self_centers, num_queries)

        return loss_center, loss_geomed, loss_sep

    def _calculate_center_loss(
        self, img_feats_masked, pts_feats_masked, centers_masked, labels, num_queries
    ):
        norms_img = torch.norm(img_feats_masked - centers_masked, p=2, dim=1) 
        norms_pts = torch.norm(pts_feats_masked - centers_masked, p=2, dim=1) 

        loss_center_perquery = norms_img + norms_pts

        loss_center = loss_center_perquery.sum()

        return loss_center / num_queries * 10

    def _calculate_geomed_loss(
        self, img_feats_masked, pts_feats_masked, centers_masked, labels, num_queries
    ):
        numerator_img = img_feats_masked - centers_masked  
        numerator_pts = pts_feats_masked - centers_masked

        denominator_img = torch.norm(numerator_img, p=2, dim=1, keepdim=True)  # [33, 1]
        denominator_pts = torch.norm(numerator_pts, p=2, dim=1, keepdim=True)

        normalized_img = numerator_img / denominator_img  
        normalized_pts = numerator_pts / denominator_pts  
        combined_terms = normalized_img + normalized_pts  

       
        assert (
            (img_feats_masked[0] - centers_masked[0]) / denominator_img[0]
            == normalized_img[0]
        ).all()
        assert (
            (pts_feats_masked[0] - centers_masked[0]) / denominator_pts[0]
            == normalized_pts[0]
        ).all()

        # Sum over queries
        sum_combined_terms = combined_terms.sum(dim=0) 
        loss_geomed = torch.norm(sum_combined_terms, p=2, dim=0) ** 2  # []: constant

        return loss_geomed / num_queries * 3
    
    def _calculate_separate_loss(self, self_centers, num_queries):
        lambda_val = 0.1
        num_classes = self_centers.size(0)
        C1 = self_centers.unsqueeze(1).expand(num_classes, num_classes, 128)
        C2 = self_centers.unsqueeze(0).expand(num_classes, num_classes, 128)
        loss_sep = -lambda_val * (C1 - C2).pow(2).sum(dim=2).triu(1).sum()
        return loss_sep / num_queries / 7
