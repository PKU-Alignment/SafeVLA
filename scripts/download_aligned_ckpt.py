import argparse
import os
from multiprocessing import Pool

from huggingface_hub import hf_hub_download


def download_ckpt(info):
    repo_id = info["repo_id"]
    filename = info["filename"]
    save_dir = info["save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    print(f"Downloading {filename} to {save_dir}...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=save_dir,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Download completed: {downloaded_path}")
    return downloaded_path


def main():
    parser = argparse.ArgumentParser(
        description="Download trained model weights from Hugging Face."
    )
    parser.add_argument(
        "--ckpt_ids",
        required=True,
        nargs="+",
        choices=["objnav", "pickup", "fetch"],
        help="The task type to download: objnav, pickup, fetch",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="The directory path to save the downloaded files.",
    )
    parser.add_argument(
        "--num", "-n", default=1, type=int, help="The number of parallel downloads."
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    repo_id = "SafetyEmbodiedAI/safety-model"

    download_args = []
    for task_type in args.ckpt_ids:
        filename = f"safe_{task_type}.pt"
        download_args.append(
            dict(
                repo_id=repo_id,
                filename=filename,
                save_dir=args.save_dir,
            )
        )

    if args.num > 1 and len(download_args) > 1:
        with Pool(min(args.num, len(download_args))) as pool:
            pool.map(download_ckpt, download_args)
    else:
        for info in download_args:
            download_ckpt(info)

    print(f"\nAll files have been successfully downloaded to: {args.save_dir}")


if __name__ == "__main__":
    main()
