import pickle
from src.utils.mask_rle_utils import coco_rle_to_masks

def read_pickle_orig_safe(filepath):
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            # Create a new dict with only needed data
            return {
                "mask_coco_rles_orig": data.get("mask_coco_rles_resized"),
                "seg_corr_list_orig": data.get("seg_corr_list"),
            }
    except Exception as e:
        raise IndexError(f"Error reading pickle file {filepath}: {str(e)}")

def unpack_segdata(filepath):
    data = read_pickle_orig_safe(filepath)
    mask_coco_rles = data["mask_coco_rles_orig"]
    seg_corr_list = data["seg_corr_list_orig"]
    
    masks = coco_rle_to_masks(mask_coco_rles)
    return masks, seg_corr_list