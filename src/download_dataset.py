from pathlib import Path
import shutil
import kagglehub

from src.config import RAW_DATASET_DIR


def download_writing_prompts_dataset():
    """
    Downloads the Kaggle Writing Prompts dataset.

    Dataset:
    https://www.kaggle.com/datasets/ratthachat/writing-prompts/data
    """

    print("=" * 80)
    print("Downloading Kaggle Writing Prompts dataset")
    print("=" * 80)

    dataset_path = kagglehub.dataset_download("ratthachat/writing-prompts")

    dataset_path = Path(dataset_path)

    print(f"Kaggle cache path: {dataset_path}")

    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for item in dataset_path.iterdir():
        destination = RAW_DATASET_DIR / item.name

        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()

        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    print(f"Dataset copied to: {RAW_DATASET_DIR}")

    print("\nFiles found:")
    for path in RAW_DATASET_DIR.rglob("*"):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    download_writing_prompts_dataset()