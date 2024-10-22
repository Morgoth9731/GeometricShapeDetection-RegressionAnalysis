import os
import cv2
import random
import numpy as np
from PIL import Image

def load_shapes(shapes_dir):
    """
    Carga las rutas de las imágenes de cada figura.
    
    :param shapes_dir: Directorio que contiene las subcarpetas de figuras.
    :return: Diccionario con el nombre de la figura como clave y una lista de rutas de imágenes como valor.
    """
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    shapes = {}
    for shape in os.listdir(shapes_dir):
        shape_path = os.path.join(shapes_dir, shape)
        if os.path.isdir(shape_path) and shape != 'mixed':
            shapes[shape] = [os.path.join(shape_path, img) for img in os.listdir(shape_path)
                            if img.lower().endswith(supported_formats)]
            if not shapes[shape]:
                print(f"Advertencia: No se encontraron imágenes para la figura '{shape}' en {shape_path}.")
    return shapes

def check_overlap(new_bbox, existing_bboxes, buffer=10):
    """
    Verifica si la nueva caja delimitadora (bounding box) se superpone con alguna existente.
    
    :param new_bbox: Tupla (x, y, w, h) de la nueva figura.
    :param existing_bboxes: Lista de tuplas (x, y, w, h) de figuras ya colocadas.
    :param buffer: Espacio adicional entre figuras para evitar superposición.
    :return: Booleano indicando si hay superposición.
    """
    x_new, y_new, w_new, h_new = new_bbox
    for bbox in existing_bboxes:
        x, y, w, h = bbox
        if (x_new < x + w + buffer and
            x_new + w_new + buffer > x and
            y_new < y + h + buffer and
            y_new + h_new + buffer > y):
            return True
    return False

def combine_shapes(shapes_dir, mixed_dir, num_images=10, max_shapes=4, image_size=(800, 600)):
    """
    Combina figuras en imágenes con fondo blanco sin superposición.
    
    :param shapes_dir: Directorio que contiene las subcarpetas de figuras.
    :param mixed_dir: Directorio donde se guardarán las imágenes combinadas.
    :param num_images: Número de imágenes combinadas a generar.
    :param max_shapes: Número máximo de figuras por imagen.
    :param image_size: Tamaño de la imagen final (ancho, alto).
    """
    if not os.path.exists(mixed_dir):
        os.makedirs(mixed_dir)
    
    shapes = load_shapes(shapes_dir)
    
    if not shapes:
        print("No se encontraron imágenes de figuras para combinar.")
        return
    
    for i in range(num_images):
        # Crear fondo blanco
        background = Image.new('RGB', image_size, (255, 255, 255))
        existing_bboxes = []
        shapes_present = []
        
        # Número de figuras a colocar en esta imagen
        num_shapes = random.randint(1, max_shapes)
        
        for _ in range(num_shapes):
            # Seleccionar una figura aleatoria
            shape_name = random.choice(list(shapes.keys()))
            shape_images = shapes[shape_name]
            
            if not shape_images:
                print(f"Advertencia: No hay imágenes disponibles para la figura '{shape_name}'.")
                continue
            
            shape_image_path = random.choice(shape_images)
            shape_image = Image.open(shape_image_path).convert("RGBA")
            
            # Redimensionar la figura aleatoriamente
            scale = random.uniform(0.5, 1.0)
            new_size = (int(shape_image.width * scale), int(shape_image.height * scale))
            shape_image = shape_image.resize(new_size, resample=Image.LANCZOS)
            
            # Generar posición aleatoria sin superposición
            placed = False
            attempts = 0
            max_attempts = 100
            while not placed and attempts < max_attempts:
                x = random.randint(0, image_size[0] - shape_image.width)
                y = random.randint(0, image_size[1] - shape_image.height)
                new_bbox = (x, y, shape_image.width, shape_image.height)
                if not check_overlap(new_bbox, existing_bboxes, buffer=20):
                    placed = True
                    existing_bboxes.append(new_bbox)
                    shapes_present.append(shape_name)
                attempts += 1
            if not placed:
                print(f"No se pudo colocar la figura {shape_name} después de {max_attempts} intentos.")
                continue
            
            # Superponer la figura en el fondo
            background.paste(shape_image, (x, y), shape_image)
        
        # Guardar la imagen combinada
        mixed_image_path = os.path.join(mixed_dir, f'mixed_{i}.png')
        background.save(mixed_image_path)
        print(f"Guardada imagen combinada: {mixed_image_path} con figuras: {', '.join(shapes_present)}")

if __name__ == "__main__":
    base_dir = os.getcwd()
    shapes_directory = os.path.join(base_dir, 'static', 'shapes')
    mixed_directory = os.path.join(shapes_directory, 'mixed')
    
    # Generar 20 imágenes combinadas
    combine_shapes(
        shapes_dir=shapes_directory,
        mixed_dir=mixed_directory,
        num_images=20,
        max_shapes=4,
        image_size=(800, 600)
    )