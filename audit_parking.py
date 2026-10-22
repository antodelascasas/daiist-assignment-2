
from pathlib import Path
import hashlib
import random
import cv2
import numpy as np

ROOT = Path("data/parking-space-4")
OUTPUT = Path("data/audit_output")
OUTPUT.mkdir(parents=True, exist_ok=True)

splits = ["train", "valid", "test"]
names = ["empty", "occupied"]
colors = [(0, 180, 0), (0, 0, 255)]

canvas = np.full((900, 1680, 3), 245, dtype=np.uint8)
seen_hashes = {}
duplicates = []
rng = random.Random(42)

for row, split in enumerate(splits):
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    print(f"{split}: {len(images)} images")

    # Detect exact duplicates across different splits
    for path in images:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        if digest in seen_hashes:
            old_split, old_path = seen_hashes[digest]
            if old_split != split:
                duplicates.append((str(old_path), str(path)))
        else:
            seen_hashes[digest] = (split, path)

    # Select four reproducible samples
    samples = rng.sample(images, min(4, len(images)))

    for col, path in enumerate(samples):
        img = cv2.imread(str(path))
        if img is None:
            continue

        tile = cv2.resize(img, (400, 260))
        label_path = label_dir / f"{path.stem}.txt"

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                values = line.split()
                if len(values) != 5:
                    continue

                cls = int(values[0])
                x, y, w, h = map(float, values[1:])

                x1 = int((x - w / 2) * 400)
                y1 = int((y - h / 2) * 260)
                x2 = int((x + w / 2) * 400)
                y2 = int((y + h / 2) * 260)

                cv2.rectangle(
                    tile, (x1, y1), (x2, y2),
                    colors[cls], 2
                )

                cv2.putText(
                    tile, names[cls], (max(0, x1), max(12, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, colors[cls], 1
                )

        left = col * 420 + 10
        top = row * 300 + 30

        canvas[top:top+260, left:left+400] = tile

        cv2.putText(
            canvas, f"{split}: {path.name[:35]}",
            (left, top-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 0, 0), 1
        )

output_path = OUTPUT / "sample_annotations.jpg"
cv2.imwrite(str(output_path), canvas)

print("\nExact cross-split duplicate pairs:", len(duplicates))
for a, b in duplicates[:10]:
    print("DUPLICATE:", a, "<=>", b)

print("\nPreview saved to:", output_path)
