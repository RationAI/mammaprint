import torch
from torch.nn.functional import cosine_similarity, normalize


class ContrastiveLoss(torch.nn.Module):
    """Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper."""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

    def forward(self, proj_1, proj_2):
        """Forward pass of the contrastive loss.

        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = normalize(proj_1, p=2, dim=1)
        z_j = normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float()
        mask = mask.to(similarity_matrix.device)
        denominator = mask * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        return all_losses.mean(), mask * similarity_matrix
