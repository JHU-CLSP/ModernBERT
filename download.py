import argparse
import time
from huggingface_hub import snapshot_download
import glob
import os

from datetime import datetime

def download_files(repo_id, pattern, local_dir, workers=8, token=None):
    """
    Download files from a Hugging Face repository that match a specific pattern.

    Args:
        repo_id (str): The repository ID (e.g., 'username/repo-name')
        pattern (str): File pattern to match (e.g., '*.txt', 'model/*.safetensors')
        local_dir (str): Local directory to save the files
        token (str, optional): Hugging Face authentication token for private repos
    """
    # Create the local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    for attempt in range(10):
        print(f"Downloading files from {repo_id} with pattern `{pattern}` to {local_dir} [attempt {attempt}]")
        try:
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=pattern,
                local_dir=local_dir,
                repo_type="dataset",
                max_workers=workers
            )
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < 5:
                time.sleep(10)
            else:
                # two stage backoff 
                time.sleep(120)
        else:
            # we're good
            break
    else:
        raise Exception("Never managed to download files")


def main():
    parser = argparse.ArgumentParser(description='Download files from Hugging Face Hub')
    parser.add_argument('--repo', required=True, help='Repository ID (e.g., username/repo-name)')
    parser.add_argument('--pattern', required=True, help='File pattern to match (e.g., *.txt)')
    parser.add_argument('--output', required=True, help='Local directory to save files')
    parser.add_argument('--workers', default=8, type=int, help="Download workers")
    parser.add_argument('--token', help='Hugging Face authentication token', default=None)
    parser.add_argument('--dist', action='store_true', help="Use composer dist to block on other ranks")

    args = parser.parse_args()

    from datetime import datetime

    def timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    import composer.utils.dist as dist
    import torch
    dist.initialize_dist('gpu', 300.0)
    dist.barrier()
    print(f"{timestamp()} Local rank {dist.get_local_rank()} has passed the initial barrier.")
    if dist.get_local_rank() == 0:
        download_files(args.repo, args.pattern, args.output, args.workers, args.token)
    else:
        print(f"{timestamp()} Local rank {dist.get_local_rank()} skipping download")
    # makes exit cleaner?
    print(f"{timestamp()} Local rank {dist.get_local_rank()} is waiting")
    dist.barrier()
    print(f"{timestamp()} Local rank {dist.get_local_rank()} is done waiting")
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
