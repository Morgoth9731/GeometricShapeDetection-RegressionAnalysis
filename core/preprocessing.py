import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocesa la imagen aplicando detección de bordes con Canny.
    
    :param image_path: Ruta de la imagen a preprocesar.
    :return: Imagen preprocesada con bordes detectados.
    """
    # Cargamos la imagen en escala de grises
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Verificamos si la imagen se cargó correctamente
    if img is None:
        print(f"Error loading image: {image_path}")
        return None

    # Aplicamos un suavizado (blur) para reducir el ruido
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Usamos Canny para detectar los bordes de la imagen
    edges = cv2.Canny(blurred_img, 50, 150)
    return edges

def preprocess_image_for_classification(image_path):
    """
    Preprocesa la imagen para clasificación sin usar HOG.
    
    :param image_path: Ruta de la imagen a preprocesar.
    :return: Vector de características como array de numpy.
    """
    print(f"Preprocesando imagen para clasificación: {image_path}")
    edges = preprocess_image(image_path)
    if edges is None:
        return None
    edges = edges.flatten()
    print(f"Características extraídas para: {image_path}")
    return edges