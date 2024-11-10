import os
import cv2
import numpy as np
import pandas as pd
import random
from logs.logger import default_logger

def generate_combined_images(output_dir, preprocessing_dir, num_images=1000):
    shapes = ['circle', 'square', 'triangle', 'star']
    label_names = ['image'] + shapes
    labels_list = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cargar las imágenes preprocesadas de cada figura
    shape_images = {}
    for shape in shapes:
        shape_dir = os.path.join(preprocessing_dir, shape, 'train')
        image_files = [img for img in os.listdir(shape_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images = []
        for image_name in image_files:
            image_path = os.path.join(shape_dir, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
        shape_images[shape] = images
        print(f"Cargadas {len(images)} imágenes para la figura: {shape}")
        default_logger.info(f"Cargadas {len(images)} imágenes para la figura: {shape}")

    for i in range(num_images):
        # Crear una imagen en blanco
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        num_shapes = random.randint(1, 4)
        selected_shapes = random.sample(shapes, num_shapes)
        labels = [0] * len(shapes)

        # Lista para almacenar los rectángulos ocupados
        occupied_rects = []

        for shape in selected_shapes:
            attempts = 0
            max_attempts = 50
            placed = False

            while not placed and attempts < max_attempts:
                attempts += 1

                # Seleccionar una imagen aleatoria de la figura
                shape_img = random.choice(shape_images[shape])

                # Redimensionar la figura a un tamaño aleatorio
                scale_factor = random.uniform(0.5, 1.5)
                new_size = (int(shape_img.shape[1] * scale_factor), int(shape_img.shape[0] * scale_factor))
                resized_shape = cv2.resize(shape_img, new_size, interpolation=cv2.INTER_AREA)

                # Crear una máscara donde el fondo blanco será transparente
                gray_shape = cv2.cvtColor(resized_shape, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_shape, 240, 255, cv2.THRESH_BINARY_INV)
                mask_inv = cv2.bitwise_not(mask)

                # Obtener las dimensiones
                h_shape, w_shape = resized_shape.shape[:2]

                # Generar posición aleatoria sin salir de los bordes
                max_x = img.shape[1] - w_shape
                max_y = img.shape[0] - h_shape
                if max_x <= 0 or max_y <= 0:
                    continue  # Saltar si la figura es más grande que la imagen base
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)

                # Crear el rectángulo de la nueva figura
                new_rect = [x, y, x + w_shape, y + h_shape]

                # Comprobar si el rectángulo se superpone con alguno existente
                overlap = False
                for rect in occupied_rects:
                    if not (new_rect[2] <= rect[0] or new_rect[0] >= rect[2] or
                            new_rect[3] <= rect[1] or new_rect[1] >= rect[3]):
                        overlap = True
                        break

                if not overlap:
                    roi = img[y:y+h_shape, x:x+w_shape]

                    # Combinar la figura con la imagen base utilizando la máscara
                    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                    shape_fg = cv2.bitwise_and(resized_shape, resized_shape, mask=mask)
                    dst = cv2.add(img_bg, shape_fg)
                    img[y:y+h_shape, x:x+w_shape] = dst

                    # Añadir el rectángulo a la lista de ocupados
                    occupied_rects.append(new_rect)

                    # Actualizar la etiqueta
                    idx = shapes.index(shape)
                    labels[idx] = 1

                    placed = True
                else:
                    continue

            if attempts == max_attempts:
                print(f"No se pudo colocar la figura {shape} sin superposición después de {max_attempts} intentos.")
                default_logger.warning(f"No se pudo colocar la figura {shape} sin superposición después de {max_attempts} intentos.")

        img_name = f"combined_{i}.png"
        img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(img_path, img)

        labels_list.append([img_name] + labels)

    # Guardar las etiquetas en labels.csv
    labels_df = pd.DataFrame(labels_list, columns=label_names)
    labels_df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)
    print(f"Generadas {len(labels_list)} imágenes combinadas en {output_dir}")
    print(f"Archivo de etiquetas guardado en {os.path.join(output_dir, 'labels.csv')}")

if __name__ == "__main__":
    preprocessing_dir = os.path.join(os.getcwd(), 'static', 'preprocessing_shapes')
    output_dir = os.path.join(os.getcwd(), 'static', 'shapes', 'combined')
    generate_combined_images(output_dir, preprocessing_dir, num_images=1000)