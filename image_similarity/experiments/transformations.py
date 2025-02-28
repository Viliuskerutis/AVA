import random
from PIL import Image, ImageEnhance


class ImageTransformations:
    """Applies random transformations for experimentation."""

    @staticmethod
    def apply_random_transformations(
        image,
        scale_range=(0.9, 1.1),
        rotation_range=(-2.5, 2.5),
        brightness_range=(0.75, 1.25),
    ):
        """
        Apply random scaling, rotation, and brightness adjustment.

        Args:
            image (PIL.Image): Input image.
            scale_range (tuple): Range for random scaling.
            rotation_range (tuple): Range for random rotation.
            brightness_range (tuple): Range for brightness adjustments.

        Returns:
            PIL.Image: Transformed image.
        """
        # Random scaling
        scale_factor = random.uniform(*scale_range)
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), Image.BICUBIC)

        # Random rotation
        angle = random.uniform(*rotation_range)
        image = image.rotate(angle, resample=Image.BICUBIC, expand=True)

        # Random brightness adjustment
        enhancer = ImageEnhance.Brightness(image)
        brightness_factor = random.uniform(*brightness_range)
        image = enhancer.enhance(brightness_factor)

        return image
