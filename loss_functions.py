import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = torch.tensor(margin)
    
    def forward(self, embeddings, labels, mining_mode="semi-hard"):
        device = embeddings.device
        self.margin.to(device)
        embedding_dimension = embeddings[0].shape[0]
        classes_in_batch = torch.unique(labels)
        dist_mat = torch.cdist(embeddings, embeddings)
        triplet_loss = torch.tensor(0.0, device=device, requires_grad=True) 
        number_of_triplets_mined = 0
        for c in classes_in_batch:
            ap_indices = (labels == c).nonzero()
            if embeddings[ap_indices].reshape(-1, embedding_dimension).shape[0] < 2:
                continue
            n_indices = (labels != c).nonzero()
            ap_pairs = torch.combinations(ap_indices.reshape(-1), r=2)
            for a_idx, p_idx in ap_pairs: 
                dist_ap = dist_mat[a_idx, p_idx]
                dists_an = dist_mat[a_idx, n_indices]

                if mining_mode == "semi-hard":
                    mined_negative_indices = torch.logical_and((dists_an > dist_ap), (dists_an < dist_ap + self.margin)).nonzero()
                elif mining_mode == "hard":
                    mined_negative_indices = (dists_an < dist_ap).nonzero()
                else:
                    mined_negative_indices = n_indices

                if mined_negative_indices.numel() == 0: # In case we don't mine any triplets
                    n_idx = n_indices[0] # Just add one to avoid nan
                else:
                    n_idx = mined_negative_indices[0,0]

                dist_an = dist_mat[a_idx, n_idx]
                loss = nn.functional.relu(dist_ap**2 - dist_an**2 + self.margin)
                triplet_loss = triplet_loss + loss.item()
                number_of_triplets_mined += 1

        return triplet_loss / number_of_triplets_mined
