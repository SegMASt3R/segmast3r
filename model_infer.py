# PyTorch Imports
import torch

# MASt3R/DUSt3R Core Imports
import mast3r_src.mast3r.model as mast3r_model

# Our Utils, DataLoaders, Models, Losses/Metrics
from src.models.mast3r_segfeat.diff_feature_matcher import featureMatcher
from src.models.mast3r_segfeat.diff_masked_pooling import masked_average_pooling


class MASt3RSegFeatInfer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Create a method to get dataset-specific model parameters
        model_params = self._get_model_params(cfg)

        # Initialize "encoder" with dataset-specific parameters
        # Here "encoder" refers to the entire MASt3R model (uptil cross-attention + MLP heads)
        self.encoder = mast3r_model.AsymmetricMASt3R(**model_params)

        # Select feature matcher + hardcode dustbin param for good measure
        cfg["FEATURE_MATCHER"]["SINKHORN"]["DUSTBIN_SCORE_INIT"] = 5.3937
        self.feature_matcher = featureMatcher(cfg["FEATURE_MATCHER"])

        # Freeze/unfreeze logic
        self._configure_grad()

    def _configure_grad(self):
        """
        Centralized method to configure gradient settings for model.
        """
        # Freeze the complete model
        self.encoder.requires_grad_(False)
        # Freeze the feature matcher
        self.feature_matcher.requires_grad_(False)

    def _get_model_params(self, cfg):
        """
        Generate model parameters based on dataset type.
        """
        # Base parameters as needed for scannetpp
        base_params = {
            "pos_embed": "RoPE100",
            "patch_embed_cls": "ManyAR_PatchEmbed",
            "img_size": (336, 512),
            "head_type": "catmlp+dpt",
            "output_mode": "pts3d+desc24",
            "depth_mode": ("exp", -mast3r_model.inf, mast3r_model.inf),
            "conf_mode": ("exp", 1, mast3r_model.inf),
            "enc_embed_dim": 1024,
            "enc_depth": 24,
            "enc_num_heads": 16,
            "dec_embed_dim": 768,
            "dec_depth": 12,
            "dec_num_heads": 12,
            "two_confs": True,
        }

        # Dataset-specific overrides
        dataset_type = cfg["DATASET"]["DATA_SOURCE"].lower()

        if dataset_type == "mapfree" or "hm3d":
            base_params.update(
                {
                    "patch_embed_cls": "PatchEmbedDust3R",
                    "img_size": (512, 512),
                    "desc_conf_mode": ("exp", 0, mast3r_model.inf),
                    "landscape_only": False,
                }
            )

        return base_params

    def forward(self, view1, view2):
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = (
                self.encoder._encode_symmetrized(view1, view2)
            )

            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

            pred1 = self.encoder._downstream_head(
                1, [tok.float() for tok in dec1], shape1
            )
            pred2 = self.encoder._downstream_head(
                2, [tok.float() for tok in dec2], shape2
            )

        return pred1["desc"], pred2["desc"]

    def prepare(self, device):
        """Call once before inference loop"""
        self.device = device
        self.eval()
        self.to(device)

    def infer_pair(self, img0, img1, masks0, masks1):
        assert hasattr(self, "device"), "Call model.prepare(device) before infer_pair"

        with torch.no_grad():
            B, _, H_resize, W_resize = img0.shape
            img0 = img0.to(self.device)  # (B, 3, H_resize, W_resize)
            img1 = img1.to(self.device)  # (B, 3, H_resize, W_resize)

            masks0 = masks0.to(self.device)  # (B, M, H_resize, W_resize)
            masks1 = masks1.to(self.device)  # (B, N, H_resize, W_resize)

            # Construct view 0 (reference id est img0) and view 1 (query id est img1)
            view0 = {
                "img": img0,
                "true_shape": torch.tensor(
                    [[H_resize, W_resize]], dtype=torch.int64, device=self.device
                ),
                "instance": [
                    f"view1_{i}" for i in range(B)
                ],  # NOTE: view1 is MASt3R's internal name for the reference view (our view0)
            }

            view1 = {
                "img": img1,
                "true_shape": torch.tensor(
                    [[H_resize, W_resize]], dtype=torch.int64, device=self.device
                ),
                "instance": [
                    f"view2_{i}" for i in range(B)
                ],  # NOTE: view2 is MASt3R's internal name for the query view (our view1)
            }

            # Model forward pass and get feature maps
            pred0, pred1 = self.forward(view0, view1)
            desc0 = pred0.permute(
                0, 3, 1, 2
            )  # (B, H_resize, W_resize, 24) -> (B, 24, H_resize, W_resize)
            desc1 = pred1.permute(
                0, 3, 1, 2
            )  # (B, H_resize, W_resize, 24) -> (B, 24, H_resize, W_resize)

            #  Aggregation to obtain segment features (Masked Average Pooling)
            agg_desc0 = masked_average_pooling(desc0, masks0)  # (B, 24, M)
            agg_desc1 = masked_average_pooling(desc1, masks1)  # (B, 24, N)

            log_P_dustb = self.feature_matcher(agg_desc0, agg_desc1)  # (B, M+1, N+1)
            scores = torch.exp(log_P_dustb)  # (B, M+1, N+1)

            # Build pred masks
            B, M_p1, N_p1 = scores.shape
            M = M_p1 - 1  # exclude dustbin row
            N = N_p1 - 1  # exclude dustbin column

            # Regular Row-Wise ArgMax ############################################
            # # Step 1: Get best matches for each ref mask (excluding dustbin row)
            # matching_indices = scores[:, :M, :].argmax(dim=-1)  # (B, M)

            # # Step 2: Create a mask indicating where the match is not to the dustbin
            # valid_mask = matching_indices < N  # (B, M)

            # # Initialize with -1s for unmatched
            # match_result = -1 * torch.ones(
            #     (B, M), dtype=torch.int64, device=self.device
            # )  # (B, M)
            # match_result[valid_mask] = matching_indices[valid_mask]  # (B, M)
            # ######################################################################

            # Mutual Nearest Neighbor Matching ##############################
            # 1. Forward Match (Ref -> Target)
            # Find best column j for each row i (excluding ref dustbin row)
            # shape: (B, M), values in [0, N] (where N is dustbin)
            ref_to_target = scores[:, :M, :].argmax(dim=-1)

            # 2. Backward Match (Target -> Ref)
            # Find best row i for each column j (excluding target dustbin col)
            # shape: (B, N), values in [0, M] (where M is dustbin)
            target_to_ref = scores[:, :, :N].argmax(dim=-2)

            # 3. Dustbin Check
            # Ensure the forward match is not pointing to the dustbin column
            is_not_dustbin = ref_to_target < N

            # 4. Mutual Check Implementation
            # We need to verify: target_to_ref[b, ref_to_target[b, i]] == i

            # Safe Gather: ref_to_target contains indices up to N (dustbin).
            # target_to_ref only has size N. Accessing index N would cause OOB.
            # We clamp indices to N-1 just for the gather operation;
            # the 'is_not_dustbin' mask will filter out the invalid ones anyway.
            safe_indices = ref_to_target.clamp(max=N - 1)

            # Gather the reverse choice for every forward choice
            # shape: (B, M)
            reciprocal_choice = torch.gather(target_to_ref, 1, safe_indices)

            # Create grid of current row indices for comparison
            # shape: (B, M)
            row_indices = torch.arange(M, device=self.device).unsqueeze(0).expand(B, M)

            # Check if the target's choice points back to the current row
            is_mutual = reciprocal_choice == row_indices

            # 5. Final Masking
            valid_mask = is_not_dustbin & is_mutual

            # Initialize result with -1
            match_result = -1 * torch.ones(
                (B, M), dtype=torch.int64, device=self.device
            )

            # Fill valid matches
            match_result[valid_mask] = ref_to_target[valid_mask]
            ######################################################################

        return match_result  # tensor of shape (B, M), values in [-1, N-1]
