import rawpy
import cv2
import numpy as np
from pathlib import Path

input_dir = Path("exp_photos")
output_dir = Path("exp_photos_jpg")
output_dir.mkdir(exist_ok=True)

for dng in input_dir.glob("*.dng"):
    with rawpy.imread(str(dng)) as raw:
        rgb = raw.postprocess()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out_path = output_dir / (dng.stem + ".jpg")
    cv2.imwrite(str(out_path), bgr)
    print(f"Converted: {dng.name} -> {out_path.name}")

print("Done.")