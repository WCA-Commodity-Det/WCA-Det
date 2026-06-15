from pathlib import Path


ROOT = Path("D:/spsb/data_merged")
IMAGE_DIR = ROOT / "images" / "train"
LABEL_DIR = ROOT / "labels" / "train"
OUT_TRAIN = ROOT / "train_weighted_simam.txt"
OUT_YAML = ROOT / "dataset_merged_weighted_simam.yaml"

# Extra repeats for hard classes from current per-class mAP50-95.
# 2=bsk1_wt, 3=bss_large, 5=hs_can, 8=nfsq
EXTRA_REPEATS = {
    2: 2,
    3: 1,
    5: 1,
    8: 2,
}

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


def image_for_label(label_path: Path) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        image_path = IMAGE_DIR / f"{label_path.stem}{suffix}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"No image found for {label_path}")


def label_class(label_path: Path) -> int:
    first_line = label_path.read_text(encoding="utf-8").splitlines()[0]
    return int(first_line.split()[0])


def main():
    weighted_images = []
    counts = {i: 0 for i in range(len(NAMES))}

    for label_path in sorted(LABEL_DIR.glob("*.txt")):
        cls = label_class(label_path)
        image_path = image_for_label(label_path)
        repeats = 1 + EXTRA_REPEATS.get(cls, 0)
        weighted_images.extend([image_path.as_posix()] * repeats)
        counts[cls] += repeats

    OUT_TRAIN.write_text("\n".join(weighted_images) + "\n", encoding="utf-8")
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

    print(f"Wrote {OUT_TRAIN} with {len(weighted_images)} weighted entries")
    print(f"Wrote {OUT_YAML}")
    for cls, count in counts.items():
        print(f"{cls:02d} {NAMES[cls]} {count}")


if __name__ == "__main__":
    main()
