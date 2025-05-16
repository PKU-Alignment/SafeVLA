import argparse
import os
from multiprocessing import Pool

from objathor.utils.download_utils import download_with_progress_bar

def download_ckpt(info):
    url = info["url"]
    save_dir = info["save_dir"]
    id = 'safevla_' + info["url"][-2:]
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir,id)
    download_with_progress_bar(
        url=url,
        save_path=ckpt_path,
        desc=f"Downloading: {url}",
    )


def main():
    parser = argparse.ArgumentParser(description="Trained ckpt downloader.")
    parser.add_argument("--save_dir", required=True, help="Directory to save the downloaded files.")
    parser.add_argument("--num", "-n", default=1, type=int, help="Number of parallel downloads.")
    args = parser.parse_args()


    os.makedirs(args.save_dir, exist_ok=True)

    download_args = []
    for id in ['aa', 'ab', 'ac']:
        save_dir = os.path.join(args.save_dir)
        download_args.append(
            dict(
                url=f"https://pub-ee94e729c6fe46f491f4f5312d417083.r2.dev/safevla_"+id,
                save_dir=save_dir,
            )
        )

    with Pool(args.num) as pool:
        pool.map(download_ckpt, download_args)


if __name__ == "__main__":
    main()
