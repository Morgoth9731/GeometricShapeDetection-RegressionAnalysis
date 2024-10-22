import sys
import os

# Añadir la carpeta raíz al PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
from core.utils import load_dataset
from core.model import (
    train_model,
    save_model,
    load_model,
)
from logs.logger import default_logger  # Importación actualizada
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, hamming_loss
from core.preprocessing import preprocess_image_for_classification  # Importación nueva
from core.preprocessing import preprocess_image  # Función existente para opción 2

def predict_figures(model, image_path, shapes, mode='single', threshold=0.5):
    """
    Realiza una predicción sobre una imagen utilizando el modelo proporcionado.
    
    :param model: Modelo entrenado.
    :param image_path: Ruta de la imagen a predecir.
    :param shapes: Lista de nombres de figuras.
    :param mode: 'single' para una figura específica, 'multi' para múltiples figuras.
    :param threshold: Umbral para decidir si una figura está presente.
    :return: Lista de predicciones.
    """
    print(f"Realizando predicción para la imagen: {image_path}")
    if mode == 'single':
        preprocess_function = preprocess_image_for_classification
    else:
        preprocess_function = preprocess_image  # Aunque no se usa en opción 2 ahora
    
    features = preprocess_function(image_path)
    if features is None:
        print(f"Error: La imagen {image_path} no pudo ser preprocesada.")
        return None
    features = features.reshape(1, -1)  # Asegurar la forma correcta
    
    # Hacer la predicción con el modelo entrenado
    prediction = model.predict(features)
    return [prediction[0]]  # Solo una figura

def visualize_prediction(image_path, prediction, shapes, mode='single', selected_shape=None):
    """
    Visualiza la imagen con las etiquetas de predicción.
    
    :param image_path: Ruta de la imagen.
    :param prediction: Lista de predicciones.
    :param shapes: Lista de nombres de figuras.
    :param mode: 'single' o 'multi'.
    :param selected_shape: Nombre de la figura seleccionada (solo para 'single').
    """
    print(f"Visualizando predicción para: {image_path}")
    # Cargar la imagen original
    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo cargar la imagen {image_path} para visualización.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crear etiquetas basadas en la predicción
    if mode == 'single' and selected_shape:
        # Solo mostrar la figura seleccionada
        present = prediction[0]
        label = f"{selected_shape.capitalize()} {'Detectada' if present else 'NO Detectada'}"
    elif mode == 'multi':
        detected = [shape for shape, present in zip(shapes, prediction) if present]
        label = "Figuras Detectadas: " + ", ".join(detected) if detected else "No se detectaron figuras"
    else:
        # Modo por defecto
        detected = [shape for shape, present in zip(shapes, prediction) if present]
        label = "Figuras Detectadas: " + ", ".join(detected) if detected else "No se detectaron figuras"
    
    # Agregar el texto en la imagen con ajustes
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img_rgb, 
        label, 
        (10, 30), 
        font, 
        0.5,                 # font_scale 
        (0, 255, 0),         # color
        1,                   # thickness
        cv2.LINE_AA
    )
    
    # Mostrar la imagen
    plt.figure(figsize=(8,6))
    plt.imshow(img_rgb)
    plt.title("Resultado de la Predicción")
    plt.axis('off')
    plt.show()

def detect_and_annotate_shapes(image_path, annotated_dir, shapes):
    """
    Detecta y anota las figuras en una imagen combinada utilizando OpenCV.
    
    :param image_path: Ruta de la imagen combinada.
    :param annotated_dir: Directorio donde se guardarán las imágenes anotadas.
    :param shapes: Lista de nombres de figuras para la clasificación.
    :return: Lista de figuras detectadas.
    """
    print(f"Procesando detección y anotación para: {image_path}")
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}.")
        return []
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbralización inversa para binarizar la imagen
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes_detected = []
    
    for cnt in contours:
        # Aproximar el contorno para simplificar la forma
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        
        # Calcular el área para filtrar pequeños contornos
        area = cv2.contourArea(cnt)
        if area < 100:  # Ajusta este umbral según tus necesidades
            continue
        
        # Identificar la forma
        shape_type = "Unidentified"
        if len(approx) == 3:
            shape_type = "triangle"
        elif len(approx) == 4:
            # Verificar si es un cuadrado o rectángulo
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape_type = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        elif len(approx) > 4:
            # Verificar si es un círculo
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * (radius ** 2)
            if abs(circle_area - area) / area < 0.2:
                shape_type = "circle"
            else:
                shape_type = "star"  # Suponiendo que las estrellas tienen más lados
        
        if shape_type != "Unidentified" and shape_type in shapes:
            shapes_detected.append(shape_type)
            # Dibujar el contorno con un color específico
            color = get_shape_color(shape_type)
            cv2.drawContours(image, [cnt], -1, color, 2)
            
            # Calcular el centro para poner la etiqueta
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            # Poner la etiqueta
            cv2.putText(image, shape_type, (cX - 30, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Crear el directorio de anotaciones si no existe
    if not os.path.exists(annotated_dir):
        os.makedirs(annotated_dir)
    
    # Guardar la imagen anotada
    annotated_image_path = os.path.join(annotated_dir, os.path.basename(image_path))
    cv2.imwrite(annotated_image_path, image)
    print(f"Imagen anotada guardada en: {annotated_image_path}")
    
    # Mostrar la imagen anotada
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,6))
    plt.imshow(img_rgb)
    plt.title(f"Detección de Figuras en {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()
    
    return shapes_detected

def get_shape_color(shape):
    """
    Asigna un color específico a cada tipo de figura.
    
    :param shape: Tipo de figura.
    :return: Tupla de color BGR.
    """
    colors = {
        "circle": (0, 255, 0),      # Verde
        "square": (255, 0, 0),      # Azul
        "triangle": (0, 0, 255),    # Rojo
        "star": (255, 165, 0),      # Naranja
        "rectangle": (128, 0, 128)  # Morado
    }
    return colors.get(shape, (0, 0, 0))  # Negro por defecto

def main():
    shapes = ['circle', 'square', 'triangle', 'star']
    base_path = os.path.join(os.getcwd(), 'static', 'shapes')
    mixed_dir = os.path.join(base_path, 'mixed')
    annotated_dir = os.path.join(base_path, 'annotated')  # Directorio para imágenes anotadas
    
    # Crear carpetas si no existen
    if not os.path.exists(mixed_dir):
        os.makedirs(mixed_dir)
    if not os.path.exists(annotated_dir):
        os.makedirs(annotated_dir)
    
    # Solicitar al usuario que seleccione una opción
    print("Selecciona una opción para detectar figuras:")
    print("1. Detectar una figura específica (circle, square, triangle, star)")
    print("2. Detectar múltiples figuras en imágenes combinadas (mixed)")
    choice = input("Ingresa 1 o 2: ").strip()
    
    if choice == '1':
        # Modo de detección de una figura específica
        print("\nOpciones de figuras:")
        for idx, shape in enumerate(shapes, start=1):
            print(f"{idx}. {shape}")
        figure_choice = input("Selecciona una figura para detectar (circle, square, triangle, star): ").strip().lower()
        
        if figure_choice not in shapes:
            default_logger.error("Figura no válida seleccionada.")
            print("Figura no válida seleccionada.")
            return
        
        # Entrenar el modelo para la figura seleccionada
        figure_folder = os.path.join(base_path, figure_choice)
        non_figure_folders = [os.path.join(base_path, shape) for shape in shapes if shape != figure_choice]
        
        default_logger.info(f"\nProcesando figura: {figure_choice}")
        print(f"\nProcesando figura: {figure_choice}")
        
        # Verificar si las carpetas existen
        if not os.path.isdir(figure_folder):
            default_logger.error(f"Carpeta para la figura '{figure_choice}' no encontrada en {figure_folder}.")
            print(f"Carpeta para la figura '{figure_choice}' no encontrada en {figure_folder}.")
            return
        for folder in non_figure_folders:
            if not os.path.isdir(folder):
                default_logger.error(f"Carpeta '{folder}' no encontrada. Asegúrate de que todas las carpetas existen.")
                print(f"Carpeta '{folder}' no encontrada. Asegúrate de que todas las carpetas existen.")
                return
        
        # Cargar el dataset usando la función original de preprocesamiento
        X, y = load_dataset(figure_folder, non_figure_folders, num_non_figure=1500)
        default_logger.info(f"Cantidad de imágenes con figura '{figure_choice}': {sum(y)}")
        default_logger.info(f"Cantidad de imágenes sin figura '{figure_choice}': {len(y) - sum(y)}")
        print(f"Cantidad de imágenes con figura '{figure_choice}': {sum(y)}")
        print(f"Cantidad de imágenes sin figura '{figure_choice}': {len(y) - sum(y)}")
        
        # Entrenar el modelo
        print("Entrenando el modelo de Regresión Logística...")
        model = train_model(X, y)
        print("Modelo entrenado exitosamente.")
        
        # Guardar el modelo
        save_model(model, figure_choice)
        print(f"Modelo guardado como '{figure_choice}_model.pkl'")
        
        # Realizar predicciones en nuevas imágenes
        # Seleccionar imágenes aleatorias de todas las carpetas de figuras
        all_folders = [os.path.join(base_path, shape) for shape in shapes]
        
        print(f"Todas las carpetas de figuras para la predicción: {all_folders}")
        default_logger.info(f"Todas las carpetas de figuras para la predicción: {all_folders}")
        
        default_logger.info("\nRealizando predicciones en nuevas imágenes...")
        print("\nRealizando predicciones en nuevas imágenes...")
        
        for i in range(10):
            # Seleccionar una carpeta aleatoria de todas las carpetas de figuras
            folder = random.choice(all_folders)
            image_files = [img for img in os.listdir(folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                default_logger.warning(f"La carpeta {folder} está vacía.")
                print(f"La carpeta {folder} está vacía.")
                continue
            image_name = random.choice(image_files)
            image_path = os.path.join(folder, image_name)
            prediction = predict_figures(model, image_path, [figure_choice], mode='single', threshold=0.5)
            if prediction is None:
                default_logger.error(f"Imagen {image_name} en carpeta {folder}: Error al cargar la imagen.")
                print(f"Imagen {image_name} en carpeta {folder}: Error al cargar la imagen.")
            else:
                present = prediction[0]
                if present:
                    default_logger.info(f"Imagen {image_name} en carpeta {folder}: {figure_choice} detectada")
                    print(f"Imagen {image_name} en carpeta {folder}: {figure_choice} detectada")
                else:
                    default_logger.info(f"Imagen {image_name} en carpeta {folder}: {figure_choice} NO detectada")
                    print(f"Imagen {image_name} en carpeta {folder}: {figure_choice} NO detectada")
            
                # Visualizar la predicción
                visualize_prediction(image_path, prediction, [figure_choice], mode='single', selected_shape=figure_choice)
    
    elif choice == '2':
        # Opción 2: Detección de múltiples figuras en imágenes combinadas (mixed) utilizando OpenCV
        print("\nIniciando detección y anotación de múltiples figuras en imágenes combinadas...\n")
        
        # Ruta para guardar imágenes anotadas
        annotated_dir = os.path.join(base_path, 'annotated')
        if not os.path.exists(annotated_dir):
            os.makedirs(annotated_dir)
        
        # Obtener todas las imágenes combinadas
        image_files = [img for img in os.listdir(mixed_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            default_logger.warning(f"La carpeta {mixed_dir} está vacía.")
            print(f"La carpeta {mixed_dir} está vacía.")
            return
        
        for image_name in image_files:
            image_path = os.path.join(mixed_dir, image_name)
            shapes_detected = detect_and_annotate_shapes(image_path, annotated_dir, shapes)
            print(f"Figuras detectadas en {image_name}: {', '.join(shapes_detected) if shapes_detected else 'Ninguna'}\n")
        
        print("Proceso de detección y anotación completado.")
    
    else:
        default_logger.error("Opción no válida seleccionada.")
        print("Opción no válida seleccionada.")
        return

if __name__ == "__main__":
    main()