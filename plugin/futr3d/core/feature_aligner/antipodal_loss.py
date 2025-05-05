import torch
import torch.nn as nn
import torch.nn.functional as F

class AntipodalLoss(nn.Module):
    """Antipodal loss.
    Img-pcd pair should be positioned antipodally.
        
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, use_gpu=True):
        super(AntipodalLoss, self).__init__()
        self.use_gpu = use_gpu
        self.cos = nn.CosineSimilarity()

    def forward(self, img_feats, pts_feats):
        """
        Args:
            img_feats (tensor): image features (aligned in unified space).
            pts_feats (tensor): point cloud features (aligned in unified space).
        """
        # norm_img_feats = F.normalize(img_feats, p=2, dim=1)
        # norm_pts_feats = F.normalize(pts_feats, p=2, dim=1)
        # cosine_similarity = torch.sum(norm_img_feats * norm_pts_feats, dim=1)
        # loss = torch.mean(cosine_similarity)

        cosine_similarity = self.cos(img_feats, pts_feats)
        loss = torch.mean(cosine_similarity)
        return loss
    

