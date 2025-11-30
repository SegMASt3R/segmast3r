import torch
import torch.nn as nn


class featureMatcher(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if cfg["TYPE"] == "Sinkhorn":
            self.matching_mat = sinkhorn(cfg["SINKHORN"])
        else:
            print("[ERROR]: feature matcher not recognized")

    def forward(self, dsc0, dsc1):
        scores = self.matching_mat(dsc0, dsc1)

        return scores


class sinkhorn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # TODO: Set descriptor_dim through config
        self.dustbin_score = nn.Parameter(torch.tensor(cfg["DUSTBIN_SCORE_INIT"]))
        self.sinkhorn_iterations = cfg["NUM_IT"]
        self.descriptor_dim = 24

    def log_sinkhorn_iterations(
        self, Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int
    ) -> torch.Tensor:
        """Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)

    def log_optimal_transport(
        self, scores: torch.Tensor, alpha: torch.Tensor, iters: int
    ) -> torch.Tensor:
        """Perform Differentiable Optimal Transport in Log-space for stability"""
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m * one).to(scores), (n * one).to(scores)

        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)

        couplings = torch.cat(
            [torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1
        )

        norm = -(ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
        Z = Z - norm  # multiply probabilities by M+N
        return Z

    def forward(self, dsc0, dsc1):
        # Compute matching descriptor distance.
        scores = torch.einsum("bdn,bdm->bnm", dsc0, dsc1)
        scores = scores / self.descriptor_dim**0.5

        scores = self.log_optimal_transport(
            scores, self.dustbin_score, iters=self.sinkhorn_iterations
        )
        # NOTE: Returning with dustbin scores (for only matching matrix, use scores[:, :-1, :-1])
        # NOTE: torch.exp(scores[:, :-1, :]) with dustbin column attached yields matching probabilities -> all rows sum to 1
        return scores
