from pathlib import Path
import shutil

import cv2
import numpy as np


ROOT = Path("D:/spsb/data_merged")
TRAIN_IMG = ROOT / "images" / "train"
TRAIN_LBL = ROOT / "labels" / "train"
AUG_IMG = ROOT / "images" / "train_weak_aug"
AUG_LBL = ROOT / "labels" / "train_weak_aug"
OUT_TRAIN = ROOT / "train_weak_aug.txt"
OUT_YAML = ROOT / "dataset_merged_weak_aug.yaml"

NAMES = [
    "bsk1",
    "bsk1_con",
    "bsk1_wt",
    "bss_large",
    "fd_can",
    "hs_can",
    "kkkl_can",
    "mnd_can",
    "nfsq",
    "xb_can",
    "xb_wt",
    "xb",
]

# Hard classes from current validation AP: bsk1_wt, bss_large, hs_can, nfsq.
# The weakest classes receive two photometric variants; medium-hard classes receive one.
WEAK_VARIANTS = {
    2: ("bright_contrast", "dim_sharp"),
    3: ("bright_contrast",),
    5: ("bright_contrast",),
    8: ("bright_contrast", "dim_sharp"),
}


def read_class(label_path: Path) -> int:
    first = label_path.read_text(encoding="utf-8").splitlines()[0]
    return int(first.split()[0])


def adjust_brightness_contrast(img: np.ndarray) -> np.ndarray:
    img = cv2.convertScaleAbs(img, alpha=1.14, beta=-6)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 1.08
    hsv[..., 2] *= 1.04
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def dim_sharpen(img: np.ndarray) -> np.ndarray:
    img = cv2.convertScaleAbs(img, alpha=0.92, beta=10)
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = cv2.addWeighted(img, 1.45, blur, -0.45, 0)
    rng = np.random.default_rng(20260604)
    noise = rng.normal(0, 3.0, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment(img: np.ndarray, variant: str) -> np.ndarray:
    if variant == "bright_contrast":
        return adjust_brightness_contrast(img)
    if variant == "dim_sharp":
        return dim_sharpen(img)
    raise ValueError(f"Unknown variant: {variant}")


def main():
    AUG_IMG.mkdir(parents=True, exist_ok=True)
    AUG_LBL.mkdir(parents=True, exist_ok=True)

    train_entries = []
    class_entries = {i: 0 for i in range(len(NAMES))}
    aug_entries = {i: 0 for i in range(len(NAMES))}

    for label_path in sorted(TRAIN_LBL.glob("*.txt")):
        cls = read_class(label_path)
        image_path = TRAIN_IMG / f"{label_path.stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        train_entries.append(image_path.as_posix())
        class_entries[cls] += 1

        variants = WEAK_VARIANTS.get(cls, ())
        if not variants:
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        for variant in variants:
            out_image = AUG_IMG / f"{label_path.stem}_{variant}.jpg"
            out_label = AUG_LBL / f"{label_path.stem}_{variant}.txt"
            aug_img = augment(img, variant)
            cv2.imwrite(str(out_image), aug_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            shutil.copyfile(label_path, out_label)
            train_entries.append(out_image.as_posix())
            class_entries[cls] += 1
            aug_entries[cls] += 1

    OUT_TRAIN.write_text("\n".join(train_entries) + "\n", encoding="utf-8")
    names_text = ", ".join(f"'{name}'" for name in NAMES)
    OUT_YAML.write_text(
        "\n".join(
            [
                f"train: {OUT_TRAIN.as_posix()}",
                f"val: {(ROOT / 'images' / 'val').as_posix()}",
                "",
                "nc: 12",
                f"names: [{names_text}]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {OUT_TRAIN} with {len(train_entries)} entries")
    print(f"Wrote {OUT_YAML}")
    for cls, name in enumerate(NAMES):
        print(f"{cls:02d} {name}: train_entries={class_entries[cls]}, augmented={aug_entries[cls]}")


if __name__ == "__main__":
    main()
