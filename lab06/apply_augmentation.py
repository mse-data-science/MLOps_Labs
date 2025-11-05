import os
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np


def apply_augmentation(image_path, save_dir, augmentation):
    image = Image.open(image_path)
    augmented = augmentation(image=np.array(image))
    augmented_image = Image.fromarray(augmented['image'])
    image_name = os.path.basename(image_path)
    augmented_image.save(os.path.join(save_dir, image_name))


def main(args):
    augmentation = getattr(A, args.augmentation)(**args.augmentation_params)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for filename in os.listdir(args.input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(args.input_dir, filename)
            apply_augmentation(image_path, args.save_dir, augmentation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Albumentations augmentations to a directory of images.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing input images.")
    parser.add_argument("save_dir", type=str, help="Path to the directory to save augmented images.")
    parser.add_argument("--augmentation", type=str, default="HorizontalFlip",
                        choices=[name for name in dir(A) if name[0].isupper()],
                        help="Name of the augmentation class.")
    parser.add_argument("--augmentation_params", type=str, nargs='*', default=[],
                        help="Parameters for the augmentation in the format key1=value1 key2=value2 ...")

    args = parser.parse_args()

    args.augmentation_params = dict(item.split('=') for item in args.augmentation_params)
    args.augmentation_params = {key: float(value) for key, value in args.augmentation_params.items()}

    main(args)
