import cv2
import numpy as np
import os
import random
import shutil
import logging

# Configuración del logger
logging.basicConfig(
    filename='logs/preprocessing.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
default_logger = logging.getLogger()

def preprocess_images(shapes_dir, preprocessing_dir, test_ratio=0.1):
    """
    Preprocesa las imágenes de figuras, aplica aumentaciones y las divide en conjuntos de entrenamiento y prueba.
    """
    # Definir colores para el cambio de color
    color_palette = [
        (255, 0, 0),      # Azul
        (0, 255, 0),      # Verde
        (0, 0, 255),      # Rojo
        (255, 165, 0),    # Naranja
        (128, 0, 128),    # Morado
        (0, 255, 255),    # Cian
        (255, 192, 203),  # Rosa
        (0, 128, 128),    # Verde Azulado
        (128, 128, 0),    # Oliva
        (128, 0, 0)       # Marrón
    ]

    shapes = [shape for shape in os.listdir(shapes_dir) if os.path.isdir(os.path.join(shapes_dir, shape))]

    for shape in shapes:
        default_logger.info(f"\nProcesando imágenes para la figura: {shape}")
        print(f"\nProcesando imágenes para la figura: {shape}")

        # Directorio de imágenes originales
        shape_dir = os.path.join(shapes_dir, shape)
        image_files = [img for img in os.listdir(shape_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Lista para almacenar todas las imágenes preprocesadas
        all_images = []

        for image_name in image_files:
            image_path = os.path.join(shape_dir, image_name)
            image = cv2.imread(image_path)
            if image is None:
                default_logger.error(f"No se pudo cargar la imagen {image_path}")
                continue

            # Imagen original redimensionada
            image_resized = cv2.resize(image, (200, 200))
            all_images.append(image_resized)

            # Rotar la imagen con un ángulo aleatorio
            random_angle = random.randint(1, 359)
            rotated_image = rotate_image(image_resized, random_angle)
            all_images.append(rotated_image)

            # Cambiar el color de la imagen original a un color aleatorio
            original_color = random.choice(color_palette)
            colored_image = change_color(image_resized, original_color)
            all_images.append(colored_image)

            # Cambiar el color de la imagen rotada a un color aleatorio diferente
            remaining_colors = [color for color in color_palette if color != original_color]
            rotated_color = random.choice(remaining_colors)
            colored_rotated_image = change_color(rotated_image, rotated_color)
            all_images.append(colored_rotated_image)

        # Mezclar las imágenes
        random.shuffle(all_images)

        # Dividir en entrenamiento y prueba
        num_images = len(all_images)
        num_test = int(num_images * test_ratio)
        test_images = all_images[:num_test]
        train_images = all_images[num_test:]

        # Crear directorios de destino
        train_dir = os.path.join(preprocessing_dir, shape, 'train')
        test_dir = os.path.join(preprocessing_dir, shape, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Guardar imágenes de entrenamiento
        for idx, img in enumerate(train_images):
            img_filename = f"{shape}_train_{idx}.png"
            img_save_path = os.path.join(train_dir, img_filename)
            cv2.imwrite(img_save_path, img)

        # Guardar imágenes de prueba
        for idx, img in enumerate(test_images):
            img_filename = f"{shape}_test_{idx}.png"
            img_save_path = os.path.join(test_dir, img_filename)
            cv2.imwrite(img_save_path, img)

        default_logger.info(f"Se han guardado {len(train_images)} imágenes en {train_dir}")
        default_logger.info(f"Se han guardado {len(test_images)} imágenes en {test_dir}")
        print(f"Se han guardado {len(train_images)} imágenes en {train_dir}")
        print(f"Se han guardado {len(test_images)} imágenes en {test_dir}")

    default_logger.info("\nPreprocesamiento y división de datos completado para todas las figuras.")
    print("\nPreprocesamiento y división de datos completado para todas las figuras.")

def rotate_image(image, angle):
    """
    Rota una imagen alrededor de su centro.
    """
    default_logger.info(f"Rotando imagen {angle} grados.")
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Matriz de rotación
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotar la imagen
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return rotated

def change_color(image, new_color):
    """
    Cambia el color de relleno de una figura negra a un nuevo color.
    """
    default_logger.info("Cambiando color de la imagen.")
    # Convertir a color si está en escala de grises
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    # Crear una máscara donde el color es negro
    mask = cv2.inRange(image_color, (0, 0, 0), (50, 50, 50))

    # Cambiar el color de los píxeles negros a new_color
    image_color[mask == 255] = new_color

    return image_color

def preprocess_image_for_classification(image_path, size=(200, 200)):
    """
    Preprocesa la imagen para clasificación:
    - Convertir a escala de grises.
    - Redimensionar a un tamaño fijo.
    - Aplicar detección de bordes.
    - Aplana la imagen a un vector de características.
    """
    default_logger.info(f"Preprocesando imagen para clasificación: {image_path}")
    try:
        # Cargar la imagen en escala de grises
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            default_logger.error(f"No se pudo cargar la imagen {image_path}")
            return None

        # Redimensionar la imagen
        img_resized = cv2.resize(img, size)

        # Aplicar suavizado (blur) para reducir ruido
        blurred_img = cv2.GaussianBlur(img_resized, (5, 5), 0)

        # Aplicar detección de bordes con Canny
        edges = cv2.Canny(blurred_img, 50, 150)

        # Aplanar la imagen
        features = edges.flatten()
        default_logger.info(f"Características extraídas para: {image_path} con {features.size} características.")

        # Normalizar las características
        features = features / 255.0

        return features
    except Exception as e:
        default_logger.error(f"Error en preprocesamiento de imagen: {e}")
        return None

def preprocess_image_for_classification_array(image_array, size=(200, 200)):
    """
    Preprocesa una imagen en formato array (numpy array) para la clasificación:
    - Convertir a escala de grises.
    - Redimensionar a un tamaño fijo.
    - Aplicar detección de bordes.
    - Aplana la imagen a un vector de características.
    """
    default_logger.info("Preprocesando imagen para clasificación desde array.")
    try:
        # Convertir a escala de grises si es necesario
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            img_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image_array

        # Redimensionar la imagen
        img_resized = cv2.resize(img_gray, size)

        # Aplicar suavizado (blur) para reducir ruido
        blurred_img = cv2.GaussianBlur(img_resized, (5, 5), 0)

        # Aplicar detección de bordes con Canny
        edges = cv2.Canny(blurred_img, 50, 150)

        # Aplanar la imagen
        features = edges.flatten()
        default_logger.info(f"Características extraídas con {features.size} características.")

        # Normalizar las características
        features = features / 255.0

        return features
    except Exception as e:
        default_logger.error(f"Error en preprocesamiento de imagen desde array: {e}")
        return None