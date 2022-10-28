import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = torch.tensor(margin)
    
    def forward(self, embeddings, labels, negative_policy="semi-hard", positive_policy="easy"):
        device = embeddings.device
        self.margin.to(device)
        embedding_dimension = embeddings[0].shape[0]
        classes_in_batch = torch.unique(labels)
        dist_mat = torch.cdist(embeddings, embeddings)**2
        triplet_loss = torch.tensor(0.0, device=device, requires_grad=True) 
        number_of_triplets_mined = 0
        for c in classes_in_batch:
            ap_indices = (labels == c).nonzero()
            if embeddings[ap_indices].reshape(-1, embedding_dimension).shape[0] < 2:
                continue
            n_indices = (labels != c).nonzero()
            #ap_pairs = torch.combinations(ap_indices.reshape(-1), r=2)
            for a_idx in ap_indices: 
            #for a_idx, p_idx in ap_pairs: 
                #dist_ap = dist_mat[a_idx, p_idx]
                #dist_ap = torch.cat([dist_mat[a_idx, 0:a_idx], dist_mat[a_idx, a_idx+1:]])
                dists_an = dist_mat[a_idx, n_indices]
                
                dists_ap = dist_mat[a_idx, ap_indices]
                dists_ap = dists_ap[dists_ap.nonzero(as_tuple=True)]
                if positive_policy == "easy":
                    dist_ap = torch.min(dists_ap)
                else:
                    dist_ap = torch.max(dists_ap)

                if negative_policy == "semi-hard":
                    mined_indices = torch.logical_and((dists_an > dist_ap), (dists_an < dist_ap + self.margin)).nonzero()
                else: # Mine hard triplets
                    mined_indices = (dists_an < dist_ap).nonzero()

                if mined_indices.numel() == 0: # In case we don't mine any triplets...
                    dist_an = torch.min(dists_an)       # Just add the closest negative to avoid nan
                else:
                    dist_an = dist_mat[a_idx, mined_indices[0,0]]

                loss = nn.functional.relu(dist_ap - dist_an + self.margin)
                triplet_loss = triplet_loss + loss
                number_of_triplets_mined += 1

        return triplet_loss / number_of_triplets_mined
