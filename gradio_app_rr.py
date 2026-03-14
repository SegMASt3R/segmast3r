import gradio as gr
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import cv2
import uuid
import rerun as rr
from gradio_rerun import Rerun
from rerun.blueprint import Vertical, Horizontal, Spatial2DView, Blueprint

from configs.default import cfg
from model_infer import MASt3RSegFeatInfer
from segmentor import SegmentationPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# FASTSAM SETUP
###############################################################################
CHECKPOINT_ROOT_PATH = Path("./checkpoints/ultralytics")
fastsam_checkpoint = CHECKPOINT_ROOT_PATH / "FastSAM-x.pt"

seg_pipeline = SegmentationPipeline(fastsam_model_path=str(fastsam_checkpoint))


###############################################################################
# MAST3R SETUP
###############################################################################
cfg.merge_from_file("configs/config_eval_spp_resz.yaml")
model = MASt3RSegFeatInfer(cfg)

ckpt_path = Path("./checkpoints/segmast3r_spp.ckpt")
model.load_state_dict(
    torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"],
    strict=False,
)
model.prepare(device)


###############################################################################
# PAIRED DATASET
###############################################################################
class SimplePairedImageMaskDataset(torch.utils.data.Dataset):
    """Simplified paired dataset for single-pair inference"""

    def __init__(self, img0_np, img1_np, masks0, masks1, target=512):
        self.img0_np = img0_np
        self.img1_np = img1_np
        self.masks0 = masks0
        self.masks1 = masks1
        self.target = target

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        def resize(im):
            return cv2.resize(
                im, (self.target, self.target), interpolation=cv2.INTER_LINEAR
            )

        img0 = resize(self.img0_np)
        img1 = resize(self.img1_np)

        # Convert to CHW
        img0 = torch.from_numpy(img0).float().permute(2, 0, 1) / 255.0
        img1 = torch.from_numpy(img1).float().permute(2, 0, 1) / 255.0

        # Resize masks
        resized_masks0 = torch.nn.functional.interpolate(
            self.masks0.unsqueeze(1).float(),
            size=(self.target, self.target),
            mode="nearest",
        ).squeeze(1)

        resized_masks1 = torch.nn.functional.interpolate(
            self.masks1.unsqueeze(1).float(),
            size=(self.target, self.target),
            mode="nearest",
        ).squeeze(1)

        return {
            "img0": img0,
            "img1": img1,
            "masks0": resized_masks0.to(torch.uint8),
            "masks1": resized_masks1.to(torch.uint8),
        }


###############################################################################
# UTILITY FUNCTIONS
###############################################################################
def run_fastsam_segmentation(image_np, conf_threshold, iou_threshold, imgsz):
    """
    Run FastSAM segmentation on numpy image

    Args:
        image_np: numpy array (H, W, 3) in RGB format
        conf_threshold: confidence threshold
        iou_threshold: IoU threshold for NMS
        imgsz: input image size

    Returns:
        torch.Tensor: masks of shape (N, H, W) where N is number of instances
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = f.name
        cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    try:
        # Run FastSAM segmentation
        segments = seg_pipeline.segment(
            temp_path, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz
        )

        if segments is None or segments.masks is None:
            print("No masks generated!")
            H, W = image_np.shape[:2]
            return torch.zeros((0, H, W), dtype=torch.uint8)

        print(f"Generated {len(segments.masks)} instance masks")

        # Convert masks to torch tensor
        masks = segments.masks.data  # Already torch tensor

        # Ensure masks match image size
        H, W = image_np.shape[:2]
        if masks.shape[1:] != (H, W):
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(1).float(), size=(H, W), mode="nearest"
            ).squeeze(1)

        return masks.to(torch.uint8)

    finally:
        Path(temp_path).unlink(missing_ok=True)


def create_combined_segmentation_mask(
    masks, match_indices, original_img_shape, is_image0=True
):
    """
    Create a single segmentation image where each matched pair has the same class ID.

    Args:
        masks: torch.Tensor of shape (N, H, W) - resized masks at 512x512
        match_indices: tensor of shape (M,) where M is number of masks in image0
                      Values are indices in image1 that match, or -1 for no match
        original_img_shape: tuple (H, W) of original image dimensions
        is_image0: True if creating mask for image0, False for image1

    Returns:
        np.ndarray: segmentation image with class IDs at original resolution
    """
    H_orig, W_orig = original_img_shape
    H, W = masks.shape[1:]
    seg_image = np.zeros((H, W), dtype=np.uint16)

    if is_image0:
        # For image0: iterate through all masks in image0
        for idx0, idx1 in enumerate(match_indices):
            if idx1 >= 0:  # Only if there's a valid match
                pair_id = idx0 + 1  # Use idx0 as pair_id (1-indexed)
                mask = masks[idx0].cpu().numpy()
                seg_image[mask > 0.5] = pair_id
    else:
        # For image1: only include matched masks
        for idx0, idx1 in enumerate(match_indices):
            if idx1 >= 0:  # Only if there's a valid match
                pair_id = idx0 + 1  # Use same pair_id as image0
                mask = masks[idx1].cpu().numpy()
                seg_image[mask > 0.5] = pair_id

    # Resize back to original image dimensions
    seg_image_resized = cv2.resize(
        seg_image, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST
    )

    return seg_image_resized


###############################################################################
# MAIN PIPELINE WITH RERUN
###############################################################################
def run_full_pipeline(recording_id, upload0, upload1):
    if upload0 is None or upload1 is None:
        yield None, "Please upload two images."
        return

    # Hardcoded parameters
    conf_threshold = 0.3
    iou_threshold = 0.4
    imgsz = 768

    # Convert to RGB numpy
    img0_np = np.array(upload0)
    img1_np = np.array(upload1)

    rec = rr.RecordingStream(
        application_id="mast3r_seg_matching", recording_id=recording_id
    )
    stream = rec.binary_stream()

    # 1) FastSAM MASKS
    print("Running FastSAM on Image 1...")
    masks0 = run_fastsam_segmentation(img0_np, conf_threshold, iou_threshold, imgsz)

    print("Running FastSAM on Image 2...")
    masks1 = run_fastsam_segmentation(img1_np, conf_threshold, iou_threshold, imgsz)

    if len(masks0) == 0 or len(masks1) == 0:
        rec.log("info", rr.TextLog("No segments detected. Adjust parameters."))
        yield stream.read(), "No segments detected. Adjust parameters."
        return

    # 2) Dataset + Batch
    dataset = SimplePairedImageMaskDataset(img0_np, img1_np, masks0, masks1)
    loader = DataLoader(dataset, batch_size=1)

    batch = next(iter(loader))
    img0, img1 = batch["img0"], batch["img1"]
    m0, m1 = batch["masks0"], batch["masks1"]

    # 3) MAST3R INFERENCE
    print("Running MAST3R inference...")
    with torch.no_grad():
        match_indices = model.infer_pair(
            img0.to(device), img1.to(device), m0.to(device), m1.to(device)
        )

    # match_indices is shape (B, M) where M is number of masks in image0
    # Values are indices in image1, or -1 for no match
    match_indices_cpu = match_indices[0].detach().cpu().numpy()

    # Count valid matches (excluding -1)
    num_matches = np.sum(match_indices_cpu >= 0)

    # 4) LOG TO RERUN
    # Create annotation context with colors for each matched pair
    annotations = [(0, "background", (0, 0, 0, 0))]  # RGBA: alpha=0 => invisible

    for idx0, idx1 in enumerate(match_indices_cpu):
        if idx1 >= 0:
            pair_id = idx0 + 1
            color = tuple(np.random.randint(0, 255, 3).tolist())
            annotations.append((pair_id, f"match_{pair_id}", color))

    rec.log("/", rr.AnnotationContext(annotations), static=True)

    # Log original images
    rec.log("image0/rgb", rr.Image(img0_np))
    rec.log("image1/rgb", rr.Image(img1_np))

    # Create and log segmentation images - pass original image shapes
    seg0 = create_combined_segmentation_mask(
        m0[0], match_indices_cpu, img0_np.shape[:2], is_image0=True
    )
    seg1 = create_combined_segmentation_mask(
        m1[0], match_indices_cpu, img1_np.shape[:2], is_image0=False
    )

    rec.log("image0/segmentation", rr.SegmentationImage(seg0))
    rec.log("image1/segmentation", rr.SegmentationImage(seg1))

    # Set up blueprint
    rec.send_blueprint(
        Blueprint(
            Horizontal(
                Spatial2DView(
                    origin="image0",
                    name="Image 1",
                    contents=[
                        "image0/rgb",
                        "image0/segmentation",
                    ],
                ),
                Spatial2DView(
                    origin="image1",
                    name="Image 2",
                    contents=[
                        "image1/rgb",
                        "image1/segmentation",
                    ],
                ),
            ),
            collapse_panels=True,
        )
    )

    status_msg = f"Done! Image 1: {len(masks0)} segments, Image 2: {len(masks1)} segments, Matches: {num_matches}"
    print(status_msg)

    yield stream.read(), status_msg


def clear_viewer(recording_id):
    """Reset the viewer by creating a new recording."""
    new_recording_id = str(uuid.uuid4())
    # Return empty stream for viewer
    rec = rr.RecordingStream(
        application_id="mast3r_seg_matching", recording_id=new_recording_id
    )
    stream = rec.binary_stream()
    return new_recording_id, stream.read(), "Viewer cleared. Upload images to start."


###############################################################################
# GRADIO UI
###############################################################################
with gr.Blocks() as demo:
    gr.Markdown("## **SegMAST3R Demo with Rerun**")
    gr.Markdown(
        "Upload two images and click Run to segment with FastSAM and match segments with SegMAST3R. Results visualized in Rerun. Hover over segments to validate matches."
    )

    recording_id = gr.State(str(uuid.uuid4()))

    with gr.Row():
        img0 = gr.Image(
            label="Image 1",
            type="numpy",
            height=350,
            width=350,
            show_label=True,
            container=True,
        )
        img1 = gr.Image(
            label="Image 2",
            type="numpy",
            height=350,
            width=350,
            show_label=True,
            container=True,
        )

    with gr.Row():
        run_btn = gr.Button("Run", variant="primary")
        clear_btn = gr.Button("Clear")

    status = gr.Textbox(label="Status", interactive=False)
    viewer = Rerun(streaming=True, height=800)

    # Examples component
    gr.Examples(
        examples=[
            ["./examples/scene1_img1.jpg", "./examples/scene1_img2.jpg"],
            ["./examples/scene2_img1.jpg", "./examples/scene2_img2.jpg"],
            ["./examples/scene3_img1.jpg", "./examples/scene3_img2.jpg"],
            ["./examples/scene4_img1.jpg", "./examples/scene4_img2.jpg"],
        ],
        inputs=[img0, img1],
        label="📸 Quick Examples",
        examples_per_page=4,
    )

    run_btn.click(
        fn=run_full_pipeline,
        inputs=[recording_id, img0, img1],
        outputs=[viewer, status],
    )

    clear_btn.click(
        fn=clear_viewer,
        inputs=[recording_id],
        outputs=[recording_id, viewer, status],
    )

demo.queue()
demo.launch()
