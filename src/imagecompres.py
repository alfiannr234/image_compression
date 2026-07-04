import streamlit as st
import cv2
import numpy as np
import time
import psutil

st.set_page_config(page_title="PCA Color Image Compression", layout="wide")
"""
Styling untuk interface PCA Image Compressor
"""
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e24;
        color: white;
    }
    div.stButton > button:first-child {
        background-color: #10b981;
        color: white;
        border: none;
        border-radius: 5px;
        width: 100%;
        font-weight: bold;
        padding: 10px;
    }
    .metric-card {
        background-color: #2a2a35;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3f3f52;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

def eigen(A, num_components=30, iterations=50):
    """
    Fungsi untuk menghitung nilai eigen dan vektor eigen dari matriks kovarians
    """
    n = A.shape[0]
    eigenvalues = []
    eigenvectors = []
    A_copy = A.copy()

    for _ in range(min(num_components, n)):
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)

        for _ in range(iterations):
            v_new = np.dot(A_copy, v)
            v_new_norm = np.linalg.norm(v_new)
            if v_new_norm == 0:
                break
            v_new = v_new / v_new_norm
            v = v_new

        eigenvalue = np.dot(v.T, np.dot(A_copy, v))
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)

        A_copy = A_copy - eigenvalue * np.outer(v, v)

    return np.array(eigenvalues), np.array(eigenvectors).T


def compress(channel_matrix, k_components):
    """
    Fungsi untuk melakukan kompresi PCA pada matriks channel
    """

    A = channel_matrix.astype(np.float32)

    mean_vector = np.mean(A, axis=0)

    A_centered = A - mean_vector

    C = np.dot(A_centered.T, A_centered) / A.shape[0]

    _, eigenvectors = eigen(C, num_components=k_components)

    W = np.dot(A_centered, eigenvectors)

    A_reconstructed = np.dot(W, eigenvectors.T) + mean_vector

    return np.clip(A_reconstructed, 0, 255).astype(np.uint8)


def process_color_image_pca(img_bgr, k_components):
    """
    Fungsi untuk memproses gambar berwarna menggunakan PCA
    """
    
    max_size = 800
    h, w, c = img_bgr.shape

    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        h, w, _ = img_bgr.shape

    b_channel, g_channel, r_channel = cv2.split(img_bgr)

    b_rec = compress(b_channel, k_components)
    g_rec = compress(g_channel, k_components)
    r_rec = compress(r_channel, k_components)

    img_reconstructed_bgr = cv2.merge([b_rec, g_rec, r_rec])

    original_elements = h * w
    compressed_elements_per_channel = (h * k_components) + (k_components * w) + w
    total_compressed_elements = compressed_elements_per_channel * 3

    compression_ratio = (
        1.0 - (total_compressed_elements / (original_elements * 3))
    ) * 100

    return img_bgr, img_reconstructed_bgr, max(0.0, compression_ratio)


st.title("PCA Image Compressor")

with st.sidebar:
    st.markdown("### Pilih Foto yang Ingin Dikompresi")
    uploaded_file = st.file_uploader("Masukan Foto", type=["jpg", "jpeg", "png"])

    st.markdown("---")
    st.markdown("### Tingkat Kompresi Foto")
    k_components = st.slider(
        "Tingkat Kompresi Foto yang Diinginkan",
        min_value=1,
        max_value=100,
        value=20,
        step=1,
    )


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_input_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    start_time = time.time()
    img_ori, img_out, comp_percentage = process_color_image_pca(
        img_input_bgr, k_components
    )
    runtime = round(time.time() - start_time, 4)

    img_ori_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    col_in, col_out = st.columns(2)

    with col_in:
        st.markdown(
            "<h4 style='text-align: center;'>Gambar Input</h4>", unsafe_allow_html=True
        )
        st.image(img_ori_rgb, use_container_width=True)
        st.caption(f"Dimensi Matriks: {img_ori.shape[0]}x{img_ori.shape[1]} piksel")

    with col_out:
        st.markdown(
            "<h4 style='text-align: center;'>Gambar Output</h4>",
            unsafe_allow_html=True,
        )
        st.image(img_out_rgb, use_container_width=True)
        st.caption(f"Direkonstruksi dengan k = {k_components} arah vektor dimensi")

    st.markdown("<br>", unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            f"<div class='metric-card'><h6>Waktu Kompresi</h6><h2>{runtime} detik</h2></div>",
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"<div class='metric-card'><h6>Persentase Perbedaan Piksel Hasil Kompresi</h6><h2>{comp_percentage:.2f}%</h2></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        is_success, buffer = cv2.imencode(".png", img_out)
        if is_success:
            bin_data = buffer.tobytes()
            st.download_button(
                label="DOWNLOAD HASIL KOMPRESI",
                data=bin_data,
                file_name="hasil_kompresi_pca.png",
                mime="image/png",
                use_container_width=True,
            )
