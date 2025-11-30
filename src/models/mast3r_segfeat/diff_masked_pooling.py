import torch


def masked_average_pooling(img_feat, masks):
    """
    Parameters:
    -----------
    img_feat : torch.Tensor of shape (B, D, H, W)
        Batch of image features
    masks : torch.Tensor of shape (B, M, H, W)
        Batch of masks, M masks per image

    Returns:
    --------
    mask_enc : torch.Tensor of shape (B, D, M)
        Encoded features for each mask in each batch
    """
    # Ensure inputs are torch tensors
    if not isinstance(img_feat, torch.Tensor):
        img_feat = torch.tensor(img_feat)
    if not isinstance(masks, torch.Tensor):
        masks = torch.tensor(masks)

    masks = masks.to(dtype=img_feat.dtype)

    # Get batch and mask dimensions
    B, M = masks.shape[0], masks.shape[1]

    # Reshape masks to (B, M, N) where N = H*W
    masks = masks.view(B, M, -1)  # B, M, N

    # Reshape img_feat to (B, D, N) and transpose last 2 dims to get (B, N, D)
    img_feat = img_feat.view(B, img_feat.shape[1], -1).transpose(1, 2)  # B, N, D

    # Calculate denominator for each mask in each batch
    deno = masks.sum(dim=-1, keepdim=True)  # B, M, 1
    deno = torch.clamp(deno, min=1.0)  # Replace any zeros with ones

    # Batch matrix multiplication
    # (B, M, N) @ (B, N, D) -> (B, M, D)
    mask_enc = torch.bmm(masks, img_feat)

    # Divide by denominator with broadcasting
    mask_enc = mask_enc / deno  # B, M, D

    # Permute to final shape (B, D, M)
    mask_enc = mask_enc.permute(0, 2, 1)  # B, D, M

    return mask_enc
