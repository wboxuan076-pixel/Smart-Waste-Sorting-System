
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json

from pathlib import Path
from PIL import Image
from io import BytesIO

from image_processing import process_image


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "waste_classifier_finetuned.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Waste Sorting System",
    page_icon="♻️",
    layout="wide"
)


# ============================================================
# 3. LOAD CNN MODEL
# ============================================================

@st.cache_resource
def load_cnn_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


# ============================================================
# 4. LOAD CLASS NAMES
# ============================================================

def load_class_names():

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Format 1: ["Plastic_Bag", "Plastic_Bottle", ...]
    if isinstance(data, list):
        return [str(name) for name in data]

    # Format 2: {"0": "Plastic_Bag", "1": "Plastic_Bottle", ...}
    if isinstance(data, dict):

        if all(str(key).isdigit() for key in data.keys()):
            return [
                str(data[key])
                for key in sorted(data.keys(), key=lambda x: int(x))
            ]

        # Format 3: {"Plastic_Bag": 0, "Plastic_Bottle": 1, ...}
        if all(isinstance(value, int) for value in data.values()):
            return [
                name
                for name, index in sorted(
                    data.items(),
                    key=lambda item: item[1]
                )
            ]

    raise ValueError(
        "Unsupported class_names.json format."
    )


# ============================================================
# 5. CNN IMAGE CLASSIFICATION
# ============================================================

def classify_image(image_rgb, model, class_names):

    # Get the input dimensions expected by the model
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    image_height = input_shape[1] or 160
    image_width = input_shape[2] or 160

    # Resize the original image
    resized_image = cv2.resize(
        image_rgb,
        (image_width, image_height)
    )

    # Convert image to float32
    # The model already contains its own Rescaling layer.
    image_array = resized_image.astype(np.float32)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Predict image class
    predictions = model.predict(
        image_array,
        verbose=0
    )

    scores = predictions[0]

    if len(scores) != len(class_names):
        raise ValueError(
            "The number of model outputs does not match "
            "the number of class names."
        )

    predicted_index = int(np.argmax(scores))

    predicted_class = class_names[predicted_index]

    confidence = float(scores[predicted_index])

    return predicted_class, confidence, scores


# ============================================================
# 6. DISPLAY IMAGE PROCESSING RESULTS
# ============================================================

def display_processing_results(results):

    st.subheader("Image Processing Results")

    st.write(
        "The uploaded image is processed using several "
        "computer vision techniques."
    )

    # First row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**1. Original Image**")
        st.image(
            results["original"],
            use_container_width=True
        )

    with col2:
        st.markdown("**2. Image Enhancement (CLAHE)**")
        st.image(
            results["enhanced"],
            use_container_width=True
        )

    # Second row
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**3. HSV Visualization**")
        st.image(
            results["hsv_visualization"],
            use_container_width=True
        )

    with col4:
        st.markdown("**4. Image Segmentation (Otsu)**")
        st.image(
            results["segmentation"],
            clamp=True,
            use_container_width=True
        )

    # Third row
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**5. Morphological Opening**")
        st.image(
            results["opening"],
            clamp=True,
            use_container_width=True
        )

    with col6:
        st.markdown("**6. Morphological Closing**")
        st.image(
            results["morphology"],
            clamp=True,
            use_container_width=True
        )

    # HSV channels
    with st.expander("View HSV Channels"):

        hue_col, sat_col, val_col = st.columns(3)

        with hue_col:
            st.markdown("**Hue**")
            st.image(
                results["hue"],
                clamp=True,
                use_container_width=True
            )

        with sat_col:
            st.markdown("**Saturation**")
            st.image(
                results["saturation"],
                clamp=True,
                use_container_width=True
            )

        with val_col:
            st.markdown("**Value**")
            st.image(
                results["value"],
                clamp=True,
                use_container_width=True
            )

    st.info(
        f"Otsu Threshold Value: "
        f"{results['otsu_threshold']:.2f}"
    )


# ============================================================
# 7. MAIN APPLICATION INTERFACE
# ============================================================

def main():

    # Application title
    st.title("Smart Waste Sorting System")

    st.markdown(
        """
        ### Computer Vision-Based Plastic Waste Classification

        This system uses image processing techniques and a
        CNN-based classification model to identify different
        types of plastic waste.
        """
    )

    st.divider()

    # Sidebar
    st.sidebar.title("System Information")

    st.sidebar.markdown(
        """
        **Computer Vision Techniques**

        1. Image Enhancement
        2. Colour Space Conversion
        3. Image Segmentation
        4. Morphological Operations
        5. CNN Image Classification
        """
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        """
        **Supported Categories**

        - Plastic Bottle
        - Plastic Bag
        - Plastic Cup
        - Plastic Packaging
        """
    )

    # Image upload
    st.subheader("Step 1: Upload a Waste Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image containing one main plastic waste item."
    )

    if uploaded_file is None:

        st.info(
            "Please upload an image to start the detection process."
        )

        st.stop()

    # Read uploaded image
    try:

        image = Image.open(
            BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

        image_rgb = np.array(image)

    except Exception as error:

        st.error(f"Unable to read the uploaded image: {error}")

        st.stop()

    st.success("Image uploaded successfully!")

    st.image(
        image_rgb,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write(
        f"Image Dimensions: "
        f"{image_rgb.shape[1]} × {image_rgb.shape[0]} pixels"
    )

    st.divider()

    # Run analysis
    if st.button(
        "Analyse Image",
        type="primary",
        use_container_width=True
    ):

        try:

            # Load model and class names
            with st.spinner("Loading CNN model..."):

                model = load_cnn_model()
                class_names = load_class_names()

            # Execute image processing
            with st.spinner("Performing image processing..."):

                processing_results = process_image(image_rgb)

            # Execute CNN classification
            with st.spinner("Classifying plastic waste..."):

                predicted_class, confidence, scores = classify_image(
                    image_rgb,
                    model,
                    class_names
                )

            st.success("Image analysis completed!")

            st.divider()

            # Display image processing results
            display_processing_results(processing_results)

            st.divider()

            # Display classification result
            st.subheader("CNN Classification Result")

            st.markdown(
                f"## Predicted Category: "
                f"**{predicted_class.replace('_', ' ')}**"
            )

            st.metric(
                label="Confidence Score",
                value=f"{confidence * 100:.2f}%"
            )

            st.progress(
                max(0.0, min(1.0, confidence))
            )

            st.caption(
                "The confidence score is the model's softmax output. "
                "It is not a guaranteed probability of correctness."
            )

            # Display all class scores
            st.subheader("Class Prediction Scores")

            for index, class_name in enumerate(class_names):

                score = float(scores[index])

                st.write(
                    f"{class_name.replace('_', ' ')}: "
                    f"{score * 100:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, score))
                )

            st.divider()

            st.subheader("Analysis Summary")

            st.write(
                f"""
                - **Predicted Category:** {predicted_class.replace('_', ' ')}
                - **Confidence Score:** {confidence * 100:.2f}%
                - **Image Enhancement:** Completed
                - **HSV Conversion:** Completed
                - **Image Segmentation:** Completed
                - **Morphological Operations:** Completed
                - **CNN Classification:** Completed
                """
            )

        except Exception as error:

            st.error(
                f"An error occurred during image analysis: {error}"
            )

            st.exception(error)


# ============================================================
# 8. RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()
