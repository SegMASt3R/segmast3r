import matplotlib.pyplot as plt
import cv2
import numpy as np
import colorsys
import torch


def load_image(image_path):
    #   print(f"Loading image from {image_path}")
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def plot_images(image0, image1):
    # permute to HWC
    if image0.ndim == 4:
        image0 = image0[0].permute(1, 2, 0).cpu().numpy()
    if image1.ndim == 4:
        image1 = image1[0].permute(1, 2, 0).cpu().numpy()
    # convert to uint8
    if image0.dtype != np.uint8:
        image0 = (image0 * 255).astype(np.uint8)
    if image1.dtype != np.uint8:
        image1 = (image1 * 255).astype(np.uint8)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image0)
    plt.title("Image 0")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image1)
    plt.title("Image 1")
    plt.axis("off")

    plt.show()


def plot_masks(image0, image1, masks0, masks1):
    imgwmasks0 = superimpose_masks(image0, masks0)
    imgwmasks1 = superimpose_masks(image1, masks1)

    # show masks
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(imgwmasks0)
    plt.title("Image 0 with Masks")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(imgwmasks1)
    plt.title("Image 1 with Masks")
    plt.axis("off")
    plt.show()


def superimpose_masks(base_img, masks):
    """
    Plot colored masks with transparency on top of the base image.

    Args:
        base_img (np.ndarray): Base image in RGB format, shape (H, W, 3), dtype uint8.
        masks (torch.Tensor or dict): A dictionary or tensor where keys are object IDs (1-based)
                                      and values are binary masks (H, W).

    Returns:
        np.ndarray: The resulting image with masks overlaid.
    """
    if base_img.ndim != 3 or base_img.shape[2] != 3:
        raise ValueError("base_img must be an RGB image of shape (H, W, 3).")

    H, W, _ = base_img.shape
    vis_img = base_img.astype(np.float32)

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()  # Convert PyTorch tensor to NumPy array

    # Handle both dictionary and array-like input for masks
    if isinstance(masks, dict):
        mask_colors = _generate_distinct_colors(len(masks))
        for obj_id, mask in masks.items():
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            if mask.shape != (H, W):
                raise ValueError(
                    f"Mask for object {obj_id} must have shape ({H}, {W})."
                )
            mask = mask.astype(bool)
            color = np.array(mask_colors[obj_id - 1], dtype=np.float32) / 255.0
            vis_img = _apply_mask(vis_img, mask, color)
    else:
        N, mask_H, mask_W = masks.shape
        if (mask_H, mask_W) != (H, W):
            raise ValueError(
                "All masks must have the same spatial dimensions as the base image."
            )
        mask_colors = _generate_distinct_colors(N)
        for i in range(N):
            mask = masks[i].astype(bool)
            color = np.array(mask_colors[i], dtype=np.float32) / 255.0
            vis_img = _apply_mask(vis_img, mask, color)

    return np.clip(vis_img, 0, 255).astype(np.uint8)


def _generate_distinct_colors(num_colors):
    """
    Generate a list of visually distinct and vibrant colors using HSV color space.

    Parameters:
    - num_colors: Number of colors to generate

    Returns:
    - List of RGB color tuples, each in the range [0, 255]
    """
    colors = []
    for i in range(num_colors):
        hue = (
            i * 0.618033988749895
        ) % 1.0  # Golden ratio method for uniform color spread
        saturation = 0.8
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = tuple(int(x * 255) for x in rgb)
        colors.append(color)
    return colors


def _apply_mask(vis_img, mask, color, alpha=0.5):
    """Apply a single mask with color and transparency to the image."""
    color_mask = np.zeros_like(vis_img, dtype=np.float32)
    color_mask[mask] = color
    vis_img[mask] = vis_img[mask] * (1 - alpha) + color_mask[mask] * 255 * alpha
    return vis_img


def plot_matched_masks(image0, image1, masks0, masks1, match_indices):
    """
    Plot matched masks with consistent colors across both images.
    Unmatched masks are not displayed.

    Args:
        image0: Base image 0 (H, W, 3) or (C, H, W)
        image1: Base image 1 (H, W, 3) or (C, H, W)
        masks0: Masks for image 0, shape (M, H, W)
        masks1: Masks for image 1, shape (N, H, W)
        match_indices: Match indices, shape (M,) with values in [-1, N-1]
    """
    # Convert to numpy if needed
    if isinstance(masks0, torch.Tensor):
        masks0 = masks0.cpu().numpy()
    if isinstance(masks1, torch.Tensor):
        masks1 = masks1.cpu().numpy()
    if isinstance(match_indices, torch.Tensor):
        match_indices = match_indices.cpu().numpy()

    # Handle batch dimension
    if masks0.ndim == 4:
        masks0 = masks0[0]
    if masks1.ndim == 4:
        masks1 = masks1[0]
    if match_indices.ndim == 2:
        match_indices = match_indices[0]

    # Convert images to proper format
    if isinstance(image0, torch.Tensor):
        if image0.ndim == 4:
            image0 = image0[0]
        if image0.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            image0 = image0.permute(1, 2, 0).cpu().numpy()
            image0 = ((image0 + 1) / 2 * 255).astype(
                np.uint8
            )  # Denormalize from [-1, 1]

    if isinstance(image1, torch.Tensor):
        if image1.ndim == 4:
            image1 = image1[0]
        if image1.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            image1 = image1.permute(1, 2, 0).cpu().numpy()
            image1 = ((image1 + 1) / 2 * 255).astype(
                np.uint8
            )  # Denormalize from [-1, 1]

    # Find valid matches
    valid_mask_indices = np.where(match_indices >= 0)[0]
    num_matches = len(valid_mask_indices)

    # print(f"Found {num_matches} matches out of {len(match_indices)} masks in image 0")

    # Generate colors for all matches
    match_colors = _generate_distinct_colors(num_matches)

    # Create overlays
    vis_img0 = image0.astype(np.float32)
    vis_img1 = image1.astype(np.float32)

    for i, mask0_idx in enumerate(valid_mask_indices):
        mask1_idx = match_indices[mask0_idx]
        color = np.array(match_colors[i], dtype=np.float32) / 255.0

        # Apply mask to image 0
        mask0 = masks0[mask0_idx].astype(bool)
        vis_img0 = _apply_mask(vis_img0, mask0, color)

        # Apply corresponding mask to image 1
        mask1 = masks1[mask1_idx].astype(bool)
        vis_img1 = _apply_mask(vis_img1, mask1, color)

    vis_img0 = np.clip(vis_img0, 0, 255).astype(np.uint8)
    vis_img1 = np.clip(vis_img1, 0, 255).astype(np.uint8)

    # Plot
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(vis_img0)
    plt.title(f"Image 0 - Matched Masks ({num_matches})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(vis_img1)
    plt.title(f"Image 1 - Matched Masks ({num_matches})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return vis_img0, vis_img1


def plot_matched_masks_with_gray_background(
    image0, image1, masks0, masks1, match_indices
):
    """
    Plot matched masks with consistent colors on grayscale background.
    Only matched masks are shown in color; unmatched regions remain grayscale.

    Args:
        image0: Base image 0 (H, W, 3) or (C, H, W) or tensor
        image1: Base image 1 (H, W, 3) or (C, H, W) or tensor
        masks0: Masks for image 0, shape (B, M, H, W) or (M, H, W)
        masks1: Masks for image 1, shape (B, N, H, W) or (N, H, W)
        match_indices: Match indices, shape (B, M) or (M,) with values in [-1, N-1]

    Returns:
        tuple: (vis_img0, vis_img1) - visualized images with matched masks
    """
    # Convert to numpy if needed
    if isinstance(masks0, torch.Tensor):
        masks0 = masks0.cpu().numpy()
    if isinstance(masks1, torch.Tensor):
        masks1 = masks1.cpu().numpy()
    if isinstance(match_indices, torch.Tensor):
        match_indices = match_indices.cpu().numpy()

    # Handle batch dimension
    if masks0.ndim == 4:
        masks0 = masks0[0]
    if masks1.ndim == 4:
        masks1 = masks1[0]
    if match_indices.ndim == 2:
        match_indices = match_indices[0]

    # Convert images to proper format (H, W, 3) uint8
    def prepare_image(img):
        if isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img[0]
            if img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                img = img.permute(1, 2, 0).cpu().numpy()
                img = ((img + 1) / 2 * 255).astype(np.uint8)  # Denormalize from [-1, 1]
        return img

    image0 = prepare_image(image0)
    image1 = prepare_image(image1)

    # Convert to grayscale
    gray0 = np.dot(image0[..., :3], [0.2989, 0.5870, 0.1140])
    gray_img0 = np.stack([gray0] * 3, axis=-1).astype(np.float32)

    gray1 = np.dot(image1[..., :3], [0.2989, 0.5870, 0.1140])
    gray_img1 = np.stack([gray1] * 3, axis=-1).astype(np.float32)

    # Find valid matches
    valid_mask_indices = np.where(match_indices >= 0)[0]
    num_matches = len(valid_mask_indices)

    print(f"Found {num_matches} matches out of {len(match_indices)} masks in image 0")

    # Generate colors for all matches
    match_colors = _generate_distinct_colors(num_matches)

    # Start with grayscale base
    vis_img0 = gray_img0.copy()
    vis_img1 = gray_img1.copy()

    # Overlay only matched masks in color
    for i, mask0_idx in enumerate(valid_mask_indices):
        mask1_idx = match_indices[mask0_idx]
        color = np.array(match_colors[i], dtype=np.float32) / 255.0

        # Apply mask to image 0
        mask0 = masks0[mask0_idx].astype(bool)
        vis_img0 = _apply_mask(vis_img0, mask0, color, alpha=0.5)

        # Apply corresponding mask to image 1
        mask1 = masks1[mask1_idx].astype(bool)
        vis_img1 = _apply_mask(vis_img1, mask1, color, alpha=0.5)

    vis_img0 = np.clip(vis_img0, 0, 255).astype(np.uint8)
    vis_img1 = np.clip(vis_img1, 0, 255).astype(np.uint8)

    return vis_img0, vis_img1
