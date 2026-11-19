
import cv2
import numpy as np


# =====================================================
# 1. Image Enhancement - CLAHE
# =====================================================

def enhance_image(image_rgb):
    """
    Improve local image contrast using CLAHE.

    Input:
        image_rgb: RGB image in uint8 format.

    Output:
        enhanced_rgb: Contrast-enhanced RGB image.
    """

    # Convert RGB to LAB colour space
    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # Apply CLAHE to the L channel
    enhanced_l = clahe.apply(l_channel)

    # Merge the enhanced channels
    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    # Convert LAB back to RGB
    enhanced_rgb = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB
    )

    return enhanced_rgb


# =====================================================
# 2. Colour Space Conversion - RGB to HSV
# =====================================================

def convert_to_hsv(image_rgb):
    """
    Convert an RGB image into HSV colour space.

    Returns:
        hsv_image: HSV image.
        hsv_visualization: RGB visualization of the HSV image.
        hue: Hue channel.
        saturation: Saturation channel.
        value: Value channel.
    """

    hsv_image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2HSV
    )

    # Extract H, S and V channels
    hue, saturation, value = cv2.split(hsv_image)

    # Convert HSV back to RGB for visualization
    hsv_visualization = cv2.cvtColor(
        hsv_image,
        cv2.COLOR_HSV2RGB
    )

    return (
        hsv_image,
        hsv_visualization,
        hue,
        saturation,
        value
    )


# =====================================================
# 3. Image Segmentation - Otsu Thresholding
# =====================================================

def segment_image(image_rgb):
    """
    Perform image segmentation using Otsu thresholding.

    Returns:
        binary_mask: Binary segmentation mask.
        threshold_value: Automatically selected threshold.
    """

    # Convert RGB to grayscale
    gray_image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    # Apply a small Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(
        gray_image,
        (5, 5),
        0
    )

    # Apply Otsu's automatic thresholding
    threshold_value, binary_mask = cv2.threshold(
        blurred_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary_mask, threshold_value


# =====================================================
# 4. Morphological Operations
# =====================================================

def apply_morphology(binary_mask):
    """
    Apply morphological Opening and Closing.

    Opening:
        Removes small foreground noise.

    Closing:
        Fills small holes and connects nearby regions.
    """

    # Create an elliptical kernel
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    # Morphological Opening
    opening_result = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Morphological Closing
    closing_result = cv2.morphologyEx(
        opening_result,
        cv2.MORPH_CLOSE,
        kernel
    )

    return opening_result, closing_result


# =====================================================
# 5. Complete Image Processing Pipeline
# =====================================================

def process_image(image_rgb):
    """
    Execute the complete image processing pipeline.

    Processing steps:
        1. Image Enhancement
        2. RGB to HSV Conversion
        3. Image Segmentation
        4. Morphological Operations

    Input:
        image_rgb: RGB image as a NumPy array.

    Output:
        Dictionary containing all processing results.
    """

    # Validate input image
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            "Input image must have 3 colour channels (RGB)."
        )

    if image_rgb.dtype != np.uint8:
        raise ValueError(
            "Input image must use uint8 format."
        )

    # Step 1: Image Enhancement
    enhanced_rgb = enhance_image(image_rgb)

    # Step 2: RGB to HSV Conversion
    (
        hsv_image,
        hsv_visualization,
        hue,
        saturation,
        value
    ) = convert_to_hsv(enhanced_rgb)

    # Step 3: Image Segmentation
    binary_mask, threshold_value = segment_image(
        enhanced_rgb
    )

    # Step 4: Morphological Operations
    opening_result, closing_result = apply_morphology(
        binary_mask
    )

    # Return all processing results
    results = {
        "original": image_rgb,
        "enhanced": enhanced_rgb,
        "hsv": hsv_image,
        "hsv_visualization": hsv_visualization,
        "hue": hue,
        "saturation": saturation,
        "value": value,
        "segmentation": binary_mask,
        "opening": opening_result,
        "morphology": closing_result,
        "otsu_threshold": float(threshold_value)
    }

    return results


# =====================================================
# 6. Simple Local Test
# =====================================================

if __name__ == "__main__":
    from pathlib import Path

    # Just a test
    image_folder = Path(
        r"C:\Users\lenovo\PycharmProjects\PythonProject\TACO-master\TACO-master\data\batch_1"
    )

    # Automatically find the first image
    image_extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.png"]

    image_files = []

    for extension in image_extensions:
        image_files.extend(image_folder.glob(extension))

    if not image_files:
        print("Error: No image found in the folder.")

    else:
        image_path = image_files[0]

        print("Selected image:", image_path)

        # Read image
        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            print("Error: Cannot read image.")

        else:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB
            )

            # Execute image processing
            results = process_image(image_rgb)

            print("\nImage processing completed!")

            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    print(key, ":", value.shape)

            print("Otsu threshold:", results["otsu_threshold"])

            print("\nAll processing steps executed successfully!")