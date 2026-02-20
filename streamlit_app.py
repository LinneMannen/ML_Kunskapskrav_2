import os
import tempfile
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import mnist_preprocessor_input as preproc  # innehåller: to_mnist_like_01(path) -> (X, original, canvas)

# Behövs eftersom att min preprocess är byggd på att en bildfil skickas in.
def preprocess_pil_via_tempfile(input_pil: Image.Image):
    """
    Din preprocess tar en filväg (path: str).
    Därför:
      1) Spara PIL-bilden till en temporär PNG
      2) Kör preproc.to_mnist_like_01(tmp_path)
      3) Ta bort tempfilen
    Returnerar: (X, original_img, processed_img)
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        input_pil.convert("RGB").save(tmp_path)

    try:
        return preproc.to_mnist_like_01(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# Logik för att slippa göra prediktioner på ett tomt canvas och istället få en ruta som ber en att rita eller ladda upp en bild
def canvas_has_ink(rgba_uint8: np.ndarray, min_ink_pixels: int = 30) -> bool:
    """
    True om canvasen verkar innehålla något rit-streck.
    Vi räknar pixlar som INTE är ”nästan vita”.
    """
    rgb = rgba_uint8[:, :, :3]
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
    ink_pixels = np.count_nonzero(gray < 250)  # 250 = nästan vitt
    return ink_pixels > min_ink_pixels


# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="MNIST Digit Predictor", layout="wide")
st.header("MNIST Digit Predictor")
st.caption("Rita en siffra eller ladda upp en bild → preprocess → predict (SVC probability=True)")


# =========================
# Ladda modell
# =========================
@st.cache_resource
def load_model():
    return joblib.load("mnist_svc_production")  

model = load_model()

# =========================
# Layout
# =========================
col_left, col_right = st.columns([1.1, 1.0], gap="medium")

with col_left:
    st.subheader("Input")

    input_mode = st.radio(
        "Välj input-metod:",
        ["Rita i canvas", "Ladda upp bild"],
        horizontal=True
    )

    input_pil = None

    if input_mode == "Rita i canvas":
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=8,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=320,
            width=320,
            drawing_mode="freedraw",
            key="canvas",
        )

        if canvas_result.image_data is not None:
            rgba = canvas_result.image_data.astype("uint8")

            # Kör bara vidare om det faktiskt finns något ”bläck” i rutan
            if canvas_has_ink(rgba, min_ink_pixels=30):
                input_pil = Image.fromarray(rgba, mode="RGBA")

    else:
        uploaded = st.file_uploader("Välj en bild (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            input_pil = Image.open(uploaded).convert("RGBA")


with col_right:
    st.subheader("Results")

    if input_pil is None:
        st.info("Rita en siffra (i canvas) eller ladda upp en bild.")
        st.stop()

    # --- preprocess ---
    try:
        X, original_img, processed_img = preprocess_pil_via_tempfile(input_pil)
    except ValueError:
        st.info("Ingen siffra hittades. Rita tydligare eller ladda upp en annan bild.")
        st.stop()
    except Exception as e:
        st.error(f"Preprocess error: {e}")
        st.stop()

    # --- predict ---
    pred = int(model.predict(X)[0])

    # --- sannolikheter ---
    probs = model.predict_proba(X)[0]
    conf = float(np.max(probs))

    # Visa preprocessad 28x28
    st.markdown("### Preprocessed (MNIST-like)")
    st.image(processed_img, clamp=True, width=180)

    # Resultat
    st.markdown("### Prediction")
    st.write(f"**Digit:** {pred}")
    st.write(f"**Confidence:** {conf*100:.2f}% (predict_proba)")

    st.markdown("### Probability Distribution")
    fig = plt.figure(figsize=(6, 3))
    plt.bar(list(range(10)), probs)
    plt.xticks(range(10))
    plt.ylim(0, 1)
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    st.pyplot(fig, clear_figure=True)