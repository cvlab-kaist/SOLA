import torch


class AlignmentLoss(torch.nn.Module):
    def __init__(
        self,
        positive_weight: float = 1.0,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.positive_weight = positive_weight
        self.temperature = torch.nn.Parameter(torch.tensor(temperature))
    
    def forward(
        self,
        object_tokens: torch.Tensor,
        labels: torch.Tensor,
        pos_tokens: torch.Tensor,
        neg_tokens: torch.Tensor,
    ) -> torch.Tensor:
        
        # get labels
        n_pos, n_neg = pos_tokens.shape[1], neg_tokens.shape[1]
        assert n_pos == 1, "n_pos must be 1"
        pos_labels = labels.unsqueeze(-1).repeat(1, 1, n_pos)
        neg_labels = (1 - labels).unsqueeze(-1).repeat(1, 1, n_neg)
        
        # compute similarity matrix
        pos_logits = torch.einsum(
            "bnd,bmd->bnm",
            [object_tokens, pos_tokens],
        ) * torch.exp(self.temperature)
        neg_logits = torch.einsum(
            "bnd,bmd->bnm",
            [object_tokens, neg_tokens],
        ) * torch.exp(self.temperature)

        # mask negative logits
        neg_labels_mask = torch.zeros_like(neg_labels)
        max_indices = neg_logits.argmax(dim=-1)
        neg_labels_mask.scatter_(-1, max_indices.unsqueeze(-1), 1)
        neg_labels = neg_labels * neg_labels_mask

        # compute loss
        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=pos_logits,
            target=pos_labels,
            reduction="mean",
        )
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=neg_logits,
            target=neg_labels,
            reduction="mean",
        )

        loss = self.positive_weight * pos_loss + neg_loss

        return loss
