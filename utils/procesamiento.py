import cv2
import numpy as np
from sklearn.cluster import KMeans

def calcular_sol_sombra(imagen, umbral_manual=None):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    if umbral_manual is None:
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binaria = cv2.threshold(gris, umbral_manual, 255, cv2.THRESH_BINARY)
    blancos = np.sum(binaria == 255)
    total = binaria.size
    porcentaje_sol = blancos / total * 100
    return binaria, round(porcentaje_sol, 2)

def identificar_cluster_suelo_manual(imagen, n_clusters=3, cluster_seleccionado=2, porcentaje_recorte=0):
    alto, ancho, _ = imagen.shape
    recorte_pixeles = int(alto * porcentaje_recorte)
    imagen_recortada = imagen[recorte_pixeles:, :]

    hsv = cv2.cvtColor(imagen_recortada, cv2.COLOR_BGR2HSV)
    alto_seg, ancho_seg, _ = hsv.shape
    pixels = hsv.reshape((-1, 3))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    labels = kmeans.labels_.reshape((alto_seg, ancho_seg))

    mascara = np.uint8(labels == cluster_seleccionado) * 255
    imagen_segmentada = cv2.bitwise_and(imagen_recortada, imagen_recortada, mask=mascara)

    return imagen_segmentada, mascara

def visualizar_clusters(imagen, n_clusters=3, porcentaje_recorte=0):
    alto, ancho, _ = imagen.shape
    recorte_pixeles = int(alto * porcentaje_recorte)
    imagen_recortada = imagen[recorte_pixeles:, :]

    hsv = cv2.cvtColor(imagen_recortada, cv2.COLOR_BGR2HSV)
    alto_seg, ancho_seg, _ = hsv.shape
    pixels = hsv.reshape((-1, 3))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    labels = kmeans.labels_.reshape((alto_seg, ancho_seg))
    centers = kmeans.cluster_centers_

    resultados = []
    for i in range(n_clusters):
        mask = (labels == i).astype(np.uint8) * 255
        cluster_img = cv2.bitwise_and(imagen_recortada, imagen_recortada, mask=mask)
        h, s, v = centers[i]
        resultados.append({
            "cluster": i,
            "imagen": cluster_img,
            "color_hsv": (round(h), round(s), round(v))
        })

    return resultados