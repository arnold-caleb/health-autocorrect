import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature, device="cpu"):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, zis, zjs):
        batch_size = zis.shape[0]

        zis_norm = F.normalize(zis, dim=1)
        zjs_norm = F.normalize(zjs, dim=1)

        sim_matrix = torch.mm(zis_norm, zjs_norm.t()) / self.temperature

        positive_mask = torch.eye(batch_size, dtype=bool).to(self.device)
        negative_mask = ~positive_mask

        pos_sim = sim_matrix[positive_mask].view(batch_size, 1)
        neg_sim = sim_matrix[negative_mask].view(batch_size, -1)
        logits = torch.cat((pos_sim, neg_sim), dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)

        loss = F.cross_entropy(logits, labels)
        return loss

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, image_embeddings, text_embeddings):
        # Compute pairwise distances between embeddings
        distance_matrix = torch.cdist(image_embeddings, text_embeddings, p=2)
        
        # Get positive distances (diagonal elements)
        pos_dist = torch.diag(distance_matrix)
        
        # Compute negative distances for images
        neg_dist_image = torch.min(distance_matrix + torch.eye(distance_matrix.size(0), device=image_embeddings.device) * self.margin, dim=1)[0]
        
        # Compute negative distances for texts
        neg_dist_text = torch.min(distance_matrix + torch.eye(distance_matrix.size(0), device=image_embeddings.device) * self.margin, dim=0)[0]
        
        # Calculate triplet loss
        loss_image = torch.mean(F.relu(pos_dist - neg_dist_image + self.margin))
        loss_text = torch.mean(F.relu(pos_dist - neg_dist_text + self.margin))
        
        loss = loss_image + loss_text
        
        return loss