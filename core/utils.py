import os
import numpy as np
import random
from core.preprocessing import preprocess_image, preprocess_image_for_classification  # Importación actualizada
from logs.logger import default_logger

def load_dataset(figure_folder, non_figure_folders, num_non_figure=1500):
    """
    Carga y preprocesa las imágenes para entrenar el modelo de detección de una figura específica.
    
    :param figure_folder: Carpeta que contiene las imágenes con la figura específica.
    :param non_figure_folders: Lista de carpetas que contienen imágenes sin la figura específica.
    :param num_non_figure: Número de imágenes negativas a cargar.
    :return: Tuple (X, y) donde X son las características y y las etiquetas.
    """
    X = []
    y = []
    
    # Cargar imágenes con la figura específica
    default_logger.info(f"Cargando imágenes positivas de: {figure_folder}")
    print(f"Cargando imágenes positivas de: {figure_folder}")
    for image in os.listdir(figure_folder):
        image_path = os.path.join(figure_folder, image)
        edges = preprocess_image_for_classification(image_path)
        if edges is not None:
            edges = edges.flatten()
            X.append(edges)
            y.append(1)  # Etiqueta: 1 (con figura)
    
    # Cargar imágenes sin la figura específica
    default_logger.info(f"Cargando {num_non_figure} imágenes negativas de múltiples carpetas.")
    print(f"Cargando {num_non_figure} imágenes negativas de múltiples carpetas.")
    for i in range(num_non_figure):
        non_figure_folder = random.choice(non_figure_folders)
        image_files = os.listdir(non_figure_folder)
        if len(image_files) == 0:
            default_logger.warning(f"La carpeta {non_figure_folder} está vacía.")
            print(f"La carpeta {non_figure_folder} está vacía.")
            continue  # Evitar errores si la carpeta está vacía
        image_name = random.choice(image_files)
        image_path = os.path.join(non_figure_folder, image_name)
        edges = preprocess_image_for_classification(image_path)
        if edges is not None:
            edges = edges.flatten()
            X.append(edges)
            y.append(0)  # Etiqueta: 0 (sin figura)
    
    X = np.array(X)
    y = np.array(y)
    return X, y 

def load_dataset_multi(shapes_dir, mixed_dir, preprocess_function):
    """
    Carga y preprocesa las imágenes del dataset para detección multi-etiqueta.
    
    :param shapes_dir: Directorio que contiene las subcarpetas de figuras.
    :param mixed_dir: Directorio que contiene imágenes con múltiples figuras.
    :param preprocess_function: Función de preprocesamiento a aplicar a cada imagen.
    :return: Tuple (X, y) donde X son las características y y las etiquetas multi-etiqueta.
    """
    X = []
    y = []
    
    shapes = [shape for shape in os.listdir(shapes_dir) if os.path.isdir(os.path.join(shapes_dir, shape)) and shape != 'mixed']
    
    # Procesar imágenes individuales
    for shape in shapes:
        shape_folder = os.path.join(shapes_dir, shape)
        for image in os.listdir(shape_folder):
            image_path = os.path.join(shape_folder, image)
            features = preprocess_function(image_path)
            if features is None:
                continue
            X.append(features)
            label = {s: 0 for s in shapes}
            label[shape] = 1
            y.append([label[s] for s in shapes])
    
    # Procesar imágenes combinadas
    for image in os.listdir(mixed_dir):
        image_path = os.path.join(mixed_dir, image)
        features = preprocess_function(image_path)
        if features is None:
            continue
        X.append(features)
        # Determinar qué figuras están presentes en la imagen combinada
        label = {s: 0 for s in shapes}
        for shape in shapes:
            if shape in image.lower():
                label[shape] = 1
        y.append([label[s] for s in shapes])
    
    X = np.array(X)
    y = np.array(y)
    
    print("Forma de X:", X.shape)  # Debe ser (n_samples, características)
    print("Forma de y:", y.shape)  # Debe ser (n_samples, número de figuras)
    
    default_logger.info(f"Dataset cargado: {len(y)} muestras con {len(shapes)} etiquetas.")
    return X, y