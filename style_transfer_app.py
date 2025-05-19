import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO

# --------------------------------------------
# Image Preprocessing and Deprocessing
# --------------------------------------------
def load_and_process_image(uploaded_file, target_size=(256, 256)):
    img = Image.open(uploaded_file).convert("RGB").resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return tf.keras.applications.vgg19.preprocess_input(img)

def deprocess_image(img):
    img = img.numpy().reshape((img.shape[1], img.shape[2], 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    return np.clip(img, 0, 255).astype("uint8")

# --------------------------------------------
# Model and Loss Functions
# --------------------------------------------
def get_model():
    vgg = VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    content_layers = ["block5_conv2"]
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1"
    ]
    outputs = [vgg.get_layer(name).output for name in content_layers + style_layers]
    model = Model(inputs=vgg.input, outputs=outputs)
    return model, content_layers, style_layers

def compute_content_loss(base, target):
    return tf.reduce_mean(tf.square(base - target))

def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    vectorized = tf.reshape(tensor, [-1, channels])
    gram = tf.matmul(tf.transpose(vectorized), vectorized)
    return gram / tf.cast(tf.shape(vectorized)[0], tf.float32)

def compute_style_loss(base, target):
    return tf.reduce_mean(tf.square(gram_matrix(base) - gram_matrix(target)))

def compute_total_loss(model, content, style, combination, content_weight, style_weight):
    outputs = model(combination)
    content_outputs = outputs[:1]
    style_outputs = outputs[1:]
    content_target = model(content)[:1]
    style_targets = model(style)[1:]

    content_loss = content_weight * compute_content_loss(content_outputs[0], content_target[0])

    style_loss = 0
    for comb_out, target_style in zip(style_outputs, style_targets):
        style_loss += compute_style_loss(comb_out, target_style)
    style_loss *= style_weight / len(style_outputs)

    return content_loss + style_loss

# --------------------------------------------
# Style Transfer Function
# --------------------------------------------
def run_style_transfer(content_img_tensor, style_img_tensor, iterations=20, content_weight=1e3, style_weight=1e-2):
    model, _, _ = get_model()
    combination_image = tf.Variable(content_img_tensor, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=5.0)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            loss = compute_total_loss(model, content_img_tensor, style_img_tensor,
                                      combination_image, content_weight, style_weight)
        grads = tape.gradient(loss, combination_image)
        optimizer.apply_gradients([(grads, combination_image)])
        combination_image.assign(tf.clip_by_value(combination_image, -103.939, 255.0 - 123.68))
        yield i + 1, loss, combination_image

# --------------------------------------------
# Streamlit UI
# --------------------------------------------
st.set_page_config(layout="wide")
st.title("🎨 Neural Style Transfer App")

# Split layout into two columns: left (uploads) and right (image previews)
col_uploads, col_images = st.columns([1, 2])

with col_uploads:
    st.header("📁 Uploads")
    content_file = st.file_uploader("Content Image", type=["jpg", "jpeg", "png"])
    style_file = st.file_uploader("Style Image", type=["jpg", "jpeg", "png"])

with col_images:
    if content_file and style_file:
        st.header("🖼️ Preview")
        c1, c2 = st.columns(2)
        with c1:
            st.image(content_file, caption="Content Image", width=250)
        with c2:
            st.image(style_file, caption="Style Image", width=250)

# Sidebar controls for hyperparameters
st.sidebar.title("⚙️ Settings")
iterations = st.sidebar.slider("Iterations", 10, 100, 20, step=10)
content_weight = st.sidebar.slider("Content Weight", 100, 10000, 1000, step=100)
style_weight = st.sidebar.slider("Style Weight", 1e-4, 1.0, 1e-2, format="%.4f")

# Stylize button
if content_file and style_file and st.button("✨ Stylize"):
    with st.spinner("Running style transfer..."):
        content_tensor = load_and_process_image(content_file)
        style_tensor = load_and_process_image(style_file)

        progress = st.progress(0, text="Starting style transfer...")

        final_img = None
        for step, loss, output in run_style_transfer(content_tensor, style_tensor,
                                                     iterations, content_weight, style_weight):
            progress.progress(step / iterations, text=f"Step {step}/{iterations} | Loss: {loss.numpy():.2f}")
            final_img = output

        result = deprocess_image(final_img)
        output_image = Image.fromarray(result).resize((256, 256))

        # Center the output image
        st.markdown("---")
        st.markdown("### 🌟 Stylized Output")
        st.image(output_image, caption="Stylized Image", width=400)

        # Download button
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        st.download_button(
            label="⬇️ Download Stylized Image",
            data=img_bytes,
            file_name="stylized_output.png",
            mime="image/png"
        )
