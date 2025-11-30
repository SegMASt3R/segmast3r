"""
Lightweight Entity Segmentation Pipeline
Image -> FastSAM Segment Everything -> Visualization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import FastSAM, SAM
import time
from pathlib import Path


def measure_time(func):
    """Decorator to measure function execution time"""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.3f}s")
        return result

    return wrapper


class SegmentationPipeline:
    def __init__(
        self,
        fastsam_model_path="FastSAM-x.pt",
    ):
        """
        Initialize the segmentation pipeline with FastSAM

        Args:
            fastsam_model_path: Path to FastSAM model
        """
        print("Loading FastSAM model...")
        self.model = measure_time(FastSAM)(fastsam_model_path)

        self.last_segments = None

    @measure_time
    def segment(self, image_path, conf=0.4, iou=0.9, imgsz=1024):
        """
        Run FastSAM segmentation (segment everything)

        Args:
            image_path: Path to input image
            conf: Confidence threshold (default 0.4)
            iou: IoU threshold for NMS (default 0.9)
            imgsz: Input image size (default 1024)

        Returns:
            Segmentation results
        """
        print(f"\nSegmenting everything in {image_path}...")

        # Run FastSAM with segment everything mode
        results = self.model(
            image_path,
            device="cuda:0",
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        self.last_segments = results[0]
        print(
            f"Generated {len(results[0].masks) if results[0].masks is not None else 0} instance masks"
        )

        return results[0]

    def visualize(self, image_path, segments=None, save_path=None, show=True):
        """
        Visualize segmentation masks

        Args:
            image_path: Path to input image
            segments: Segmentation results (uses last_segments if None)
            save_path: Path to save visualization (optional)
            show: Whether to display the plot
        """
        if segments is None:
            segments = self.last_segments

        # Load image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Segmentation masks
        if segments is not None and segments.masks is not None:
            img_seg = img.copy()
            masks = segments.masks.data.cpu().numpy()

            # Create colored overlay
            overlay = np.zeros_like(img)
            for i, mask in enumerate(masks):
                # Resize mask to image size if needed
                if mask.shape != img.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8), (img.shape[1], img.shape[0])
                    )

                # Generate random color for each instance
                color = np.random.randint(0, 255, 3).tolist()
                overlay[mask > 0.5] = color

            # Blend with original image
            img_seg = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

            axes[1].imshow(img_seg)
            axes[1].set_title(f"Instance Masks ({len(masks)})")
        else:
            axes[1].imshow(img)
            axes[1].set_title("Instance Masks (0)")
        axes[1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\nSaved visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def run(self, image_path, conf=0.4, iou=0.9, imgsz=1024, save_path=None, show=True):
        """
        Complete pipeline: segment -> visualize

        Args:
            image_path: Path to input image
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Input image size
            save_path: Path to save visualization (optional)
            show: Whether to display the plot

        Returns:
            segments
        """
        segments = self.segment(image_path, conf=conf, iou=iou, imgsz=imgsz)
        self.visualize(image_path, segments, save_path, show)

        return segments


# Example usage
if __name__ == "__main__":
    CHECKPOINT_ROOT_PATH = Path("./checkpoints/ultralytics")
    fastsam_checkpoint = CHECKPOINT_ROOT_PATH / "mobile_sam.pt"

    # Initialize pipeline
    pipeline = SegmentationPipeline(
        fastsam_model_path=fastsam_checkpoint,
    )

    # Run on single image
    image_path = "./checkpoints/test_image.jpg"
    segments = pipeline.run(
        image_path,
        conf=0.6,
        iou=0.45,
        imgsz=768,
        save_path="./checkpoints/output_visualization.jpg",
        show=False,
    )

    # Or run steps individually for more control
    # segments = pipeline.segment(image_path, conf=0.4, iou=0.9, imgsz=1024)
    # pipeline.visualize(image_path, save_path='custom_output.jpg')
