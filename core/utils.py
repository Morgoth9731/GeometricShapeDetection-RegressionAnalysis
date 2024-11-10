import os
import numpy as np
from sklearn.model_selection import train_test_split
from core.preprocessing import preprocess_image_for_classification
from logs.logger import default_logger
import pandas as pd

def load_dataset(preprocessing_dir):
    """
    Carga el conjunto de datos preprocesado para el modelo de multiclasificación.
    """
    shapes = ['circle', 'square', 'triangle', 'star']
    X = []
    y = []
    label_mapping = {'circle': 0, 'square': 1, 'triangle': 2, 'star': 3}

    for shape in shapes:
        shape_train_dir = os.path.join(preprocessing_dir, shape, 'train')
        shape_test_dir = os.path.join(preprocessing_dir, shape, 'test')

        # Procesar imágenes de entrenamiento
        if os.path.exists(shape_train_dir):
            for img_name in os.listdir(shape_train_dir):
                img_path = os.path.join(shape_train_dir, img_name)
                features = preprocess_image_for_classification(img_path)
                if features is not None:
                    X.append(features)
                    y.append(label_mapping[shape])

        # Procesar imágenes de prueba
        if os.path.exists(shape_test_dir):
            for img_name in os.listdir(shape_test_dir):
                img_path = os.path.join(shape_test_dir, img_name)
                features = preprocess_image_for_classification(img_path)
                if features is not None:
                    X.append(features)
                    y.append(label_mapping[shape])

    X = np.array(X)
    y = np.array(y)

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, y_train, X_test, y_test

def load_multilabel_dataset(images_dir, labels_file, test_size=0.2):
    """
    Carga el conjunto de datos multietiqueta.
    """
    # Leer el archivo de etiquetas
    labels_df = pd.read_csv(labels_file)
    image_names = labels_df['image'].tolist()
    labels = labels_df.drop('image', axis=1).values

    X = []
    y = []

    for img_name, label in zip(image_names, labels):
        img_path = os.path.join(images_dir, img_name)
        features = preprocess_image_for_classification(img_path)
        if features is not None:
            X.append(features)
            y.append(label)
        else:
            default_logger.warning(f"No se pudo procesar la imagen {img_name}")

    X = np.array(X)
    y = np.array(y)

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, y_train, X_test, y_test