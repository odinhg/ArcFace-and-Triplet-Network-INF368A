import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, mine_hard_triplets=False, normalize_embeddings=True):
        super().__init__()
        self.margin = torch.tensor(margin)
        self._mine_hard_triplets = mine_hard_triplets
        self.normalize_embeddings = normalize_embeddings

    @property
    def mine_hard_triplets(self):
        return self._mine_hard_triplets
    
    @mine_hard_triplets.setter
    def mine_hard_triplets(self, value):
        self._mine_hard_triplets = value

    def forward(self, embeddings, labels, validation=False):
        # Mine semi-hard (and possibly hard) triplets and compute loss
        device = embeddings.device
        self.margin.to(device)
        #if self.normalize_embeddings:
        #    embeddings = nn.functional.normalize(embeddings) # Normalize embeddings
        embedding_dimension = embeddings[0].shape[0]
        classes_in_batch = torch.unique(labels)
        triplet_loss = torch.tensor(0.0, device=device) 
        number_of_triplets_mined = 0
        for c in classes_in_batch:
            same_class_samples = embeddings[(labels == c).nonzero()].reshape(-1, embedding_dimension)
            if same_class_samples.shape[0] < 2:
                continue # Need at least one distinct pair of anchor and positive 
            different_class_samples = embeddings[(labels != c).nonzero()].reshape(-1, embedding_dimension)
            for a_idx in range(0, same_class_samples.shape[0]):
                for p_idx in [i for i in range(0, same_class_samples.shape[0]) if i != a_idx]:
                    a = same_class_samples[a_idx]
                    p = same_class_samples[p_idx]
                    dist_ap = torch.sum((a-p)**2)
                    for n_idx in range(0, different_class_samples.shape[0]):
                        n = different_class_samples[n_idx]
                        dist_an = torch.sum((a-n)**2)
                        if validation:
                            triplet_loss += torch.max((dist_ap - dist_an + self.margin), torch.tensor(0)) # max not really needed here
                            number_of_triplets_mined += 1
                            break 
                        # Semi-hard triplets where d(a,p) < d(a,n) but the negative lies within the margin
                        is_semi_hard = (dist_ap < dist_an < dist_ap + self.margin)
                        # Hard triplets where d(a,n) < d(a,p)
                        is_hard = (dist_an < dist_ap)
                        if (is_semi_hard and not self.mine_hard_triplets) or (is_hard and self.mine_hard_triplets):
                            triplet_loss += torch.max((dist_ap - dist_an + self.margin), torch.tensor(0)) # max not really needed here
                            number_of_triplets_mined += 1
                            break
                    if number_of_triplets_mined >= 200:
                        return triplet_loss / number_of_triplets_mined

        triplet_loss = triplet_loss / number_of_triplets_mined
        return triplet_loss
