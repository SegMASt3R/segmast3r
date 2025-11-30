from pathlib import Path
import pytz
from datetime import datetime

import json
import msgpack


def load_metrics_auto(file_path):
    ext = "".join(file_path.suffixes)
    if ext == ".json":
        with open(file_path, "r") as f:
            return json.load(f)
    elif ext == ".msgpack":
        with open(file_path, "rb") as f:
            return msgpack.unpackb(f.read(), raw=False)
    elif ext == ".msgpack.lz4":
        import lz4.frame

        with lz4.frame.open(file_path, "rb") as f:
            return msgpack.unpackb(f.read(), raw=False)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def save_metrics_auto(file_path, data):
    ext = "".join(file_path.suffixes)
    if ext == ".json":
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    elif ext == ".msgpack":
        with open(file_path, "wb") as f:
            packed = msgpack.packb(data, use_bin_type=True)
            f.write(packed)
    elif ext == ".msgpack.lz4":
        import lz4.frame

        with lz4.frame.open(file_path, "wb") as f:
            packed = msgpack.packb(data, use_bin_type=True)
            f.write(packed)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def get_month_day():
    # Get current time in IST
    ist_time = datetime.now(pytz.timezone("Asia/Kolkata"))

    # Format as compact month name + date with 2 digits
    compact_date = ist_time.strftime("%b%d")

    print(compact_date)  # Output example: Mar09


def extract_month_day(experiment_name):
    # Split the string by underscore and get the date part
    parts = experiment_name.split("_")
    date_str = parts[-2]  # This will be "2025-03-03" from your example

    # Parse the date string
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    # Format to get compact month name + day
    compact_monthday = date_obj.strftime("%b%d")

    return compact_monthday


def build_exp_folder(args, cfg=None):
    # create results/experiment folder
    exp_name = args.experiment
    results_dir = Path(args.results) / exp_name

    cfg.SAVE_DIR = str(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    if cfg is not None:
        # save the current config
        cfg_file = results_dir / "config.yaml"
        cfg.dump(stream=open(cfg_file, "w"))
        print(f"Config saved at: {cfg_file}")
