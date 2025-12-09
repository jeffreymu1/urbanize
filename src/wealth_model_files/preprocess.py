import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("../data/PP2/final_photo_dataset")
OUTPUT_DIR = Path("../data/preprocessed_images/")
CROP_SIZE = 300

def crop_center(image, crop_size):
    width, height = image.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    return image.crop((left, top, right, bottom))

def preprocess_images():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for img_file in tqdm(list(INPUT_DIR.glob("*.jpg")), desc="Preprocessing images"):
        try:
            img = Image.open(img_file).convert("RGB")
            cropped = crop_center(img, CROP_SIZE)

            out_path = OUTPUT_DIR / img_file.name
            cropped.save(out_path, format="JPEG", quality=95)

        except Exception as e:
            print(f"Error on {img_file}: {e}")
    print("Preprocessing complete.")
if __name__ == "__main__":
    preprocess_images()