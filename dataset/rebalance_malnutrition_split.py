from pathlib import Path
import random
import shutil


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "malnutrition_dataset"
CLASSES = ("malnutrition", "normal")
SEED = 42
TRAIN_RATIO = 0.8


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path):
    ensure_dir(path)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def unique_destination(dest_dir: Path, filename: str):
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def main():
    random.seed(SEED)

    for class_name in CLASSES:
        train_dir = DATASET_DIR / "train" / class_name
        test_dir = DATASET_DIR / "test" / class_name

        ensure_dir(train_dir)
        ensure_dir(test_dir)

        all_files = list_images(train_dir) + list_images(test_dir)
        random.shuffle(all_files)

        total = len(all_files)
        train_count = int(total * TRAIN_RATIO)
        test_count = total - train_count

        temp_dir = DATASET_DIR / "_tmp_rebalance" / class_name
        clear_dir(temp_dir)

        for src in all_files:
            shutil.move(str(src), str(unique_destination(temp_dir, src.name)))

        clear_dir(train_dir)
        clear_dir(test_dir)

        temp_files = list_images(temp_dir)
        random.shuffle(temp_files)
        train_files = temp_files[:train_count]
        test_files = temp_files[train_count:]

        for src in train_files:
            shutil.move(str(src), str(unique_destination(train_dir, src.name)))
        for src in test_files:
            shutil.move(str(src), str(unique_destination(test_dir, src.name)))

        print(
            f"{class_name}: total={total} train={train_count} test={test_count}"
        )

    tmp_root = DATASET_DIR / "_tmp_rebalance"
    if tmp_root.exists():
        for class_dir in tmp_root.iterdir():
            if class_dir.is_dir() and not any(class_dir.iterdir()):
                class_dir.rmdir()
        if not any(tmp_root.iterdir()):
            tmp_root.rmdir()

    print("Rebalance complete.")


if __name__ == "__main__":
    main()
