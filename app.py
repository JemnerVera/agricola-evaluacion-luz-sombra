import streamlit as st
import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO
from utils.procesamiento import (
    calcular_sol_sombra,
    identificar_cluster_suelo_manual,
    visualizar_clusters
)

def pil_to_base64(img_array):
    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

st.set_page_config(page_title="Evaluación Luz-Sombra - Agrícola Andrea", layout="centered")
st.title("Evaluación Luz-Sombra - Agrícola Andrea")

archivo = st.file_uploader("📷 Sube una imagen agrícola", type=["jpg", "jpeg", "png"])
modo_manual = st.checkbox("Ajuste manual del umbral binario")

if archivo:
    imagen_bytes = archivo.read()
    npimg = np.frombuffer(imagen_bytes, np.uint8)
    imagen = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    st.subheader("Selecciona el método de análisis")
    modo = st.radio("", ["Segmentación automática (KMeans)", "Visualizar clusters"])

    porcentaje_recorte = st.slider("Recorte superior de la imagen (%)", 0, 50, 10, step=10) / 100.0

    st.markdown("""
    **¿Para qué sirve el recorte superior?**  
    El recorte elimina la parte superior de la foto antes del análisis, descartando cielo, hojas y troncos que no deben influir en la evaluación del suelo.
    """)

    n_clusters = 3  # fijo

    st.subheader("Selección de clusters")
    cluster_suelo = st.slider(
        "Selecciona el cluster que representa el suelo",
        min_value=0,
        max_value=2,
        value=2
    )

    st.markdown("""
    El sistema agrupa la imagen en tres zonas según color:
    - **Cluster 0**: zonas más dominantes, puede incluir suelo iluminado  
    - **Cluster 1**: hojas, cielo u otras estructuras  
    - **Cluster 2**: áreas oscuras como troncos o uvas
    """)

    if modo == "Segmentación automática (KMeans)":
        segmentada, mascara = identificar_cluster_suelo_manual(
            imagen, n_clusters, cluster_suelo, porcentaje_recorte
        )
        fuente = segmentada

        if modo_manual:
            umbral = st.slider("Umbral manual", 0, 255, 128)
            binaria, porcentaje = calcular_sol_sombra(fuente, umbral_manual=umbral)
        else:
            binaria, porcentaje = calcular_sol_sombra(fuente)

        img_original = pil_to_base64(imagen)
        img_binaria = pil_to_base64(cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR))

        st.markdown("## Evaluación sol-sombra")

        fundo = st.text_input("Fundo:")
        lote = st.text_input("Lote:")
        hilera = st.text_input("Hilera:")

        ficha_html = f"""
        <div style="border:1px solid #ccc; padding:20px; border-radius:5px; background-color:#f9f9f9">
            <h2 style="margin-bottom:5px;">Evaluación sol-sombra</h2>
            <p><strong>Fundo:</strong> {fundo} &nbsp;&nbsp; <strong>Lote:</strong> {lote} &nbsp;&nbsp; <strong>Hilera:</strong> {hilera}</p>
            <table style="width:100%; margin-top:10px;">
              <tr>
                <td style="width:50%; vertical-align:top;">
                  <img src="data:image/jpeg;base64,{img_original}" style="width:100%; border:1px solid #aaa;" />
                  <p style="text-align:center;">Imagen original</p>
                </td>
                <td style="width:50%; vertical-align:top;">
                  <img src="data:image/jpeg;base64,{img_binaria}" style="width:100%; border:1px solid #aaa;" />
                  <p style="text-align:center;">Binaria del suelo</p>
                </td>
              </tr>
            </table>
            <p style="margin-top:15px;"><strong>Porcentaje de luz sobre zona analizada:</strong> {porcentaje} %</p>
        </div>
        """
        st.markdown(ficha_html, unsafe_allow_html=True)
        st.info("📌 Para exportar la ficha como PNG, usa una captura de pantalla desde el navegador.")

    elif modo == "Visualizar clusters":
        st.subheader("Vista previa de los clusters")
        resultados = visualizar_clusters(imagen, n_clusters, porcentaje_recorte)
        for r in resultados:
            st.image(r["imagen"], caption=f"Cluster {r['cluster']} – HSV: {r['color_hsv']}", channels="BGR")

else:
    st.info("💡 Sube una imagen para comenzar la evaluación.")