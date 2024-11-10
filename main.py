import sys
import os
import subprocess  
import pickle 

# Añadir la carpeta raíz al PYTHONPATHs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core.utils import load_dataset, load_multilabel_dataset
from core.model import (
    train_model,
    train_multilabel_model,
    save_model,
    load_model,
)
from core.preprocessing import (
    preprocess_images,
    preprocess_image_for_classification,
    preprocess_image_for_classification_array
)
from logs.logger import default_logger
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    hamming_loss,          # Importación añadida
    jaccard_score
)
from sklearn.preprocessing import label_binarize

def create_graphs_subdir(subdir_name):
    """
    Crea un subdirectorio dentro de 'static/graphs' si no existe.

    Args:
        subdir_name (str): Nombre del subdirectorio a crear.

    Returns:
        str: Ruta completa del subdirectorio.
    """
    graphs_dir = os.path.join(os.getcwd(), 'static', 'graphs', subdir_name)
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)
    return graphs_dir

def plot_confusion_matrix_custom(y_true, y_pred, labels, title, save_path):
    """
    Genera y guarda una matriz de confusión.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    default_logger.info(f"Matriz de confusión guardada en: {save_path}")

def plot_classification_report_custom(y_true, y_pred, labels, title, save_path):
    """
    Genera y guarda un reporte de clasificación como un gráfico de barras.
    """
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().drop(['support'], axis=1)
    
    plt.figure(figsize=(10,8))
    report_df.iloc[:-3, :].plot(kind='bar', figsize=(10,8))
    plt.title(title)
    plt.ylabel('Score')
    plt.ylim(0,1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    default_logger.info(f"Reporte de clasificación guardado en: {save_path}")

def plot_multiclass_roc(y_true, y_pred_proba, classes, save_path):
    """
    Genera y guarda las curvas ROC para cada clase en un modelo multiclasificación.
    """
    # Binarizar las etiquetas
    y_true_binarized = label_binarize(y_true, classes=range(len(classes)))
    n_classes = y_true_binarized.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10,8))
    colors = sns.color_palette("bright", n_classes)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve de la clase {0} (AUC = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC para Multiclasificación')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    default_logger.info(f"Curvas ROC guardadas en: {save_path}")

def plot_multiclass_precision_recall(y_true, y_pred_proba, classes, save_path):
    """
    Genera y guarda las curvas de Precisión-Recall para cada clase en un modelo multiclasificación.
    """
    # Binarizar las etiquetas
    y_true_binarized = label_binarize(y_true, classes=range(len(classes)))
    n_classes = y_true_binarized.shape[1]
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_pred_proba[:, i])
        average_precision[i] = average_precision_score(y_true_binarized[:, i], y_pred_proba[:, i])
    
    plt.figure(figsize=(10,8))
    colors = sns.color_palette("bright", n_classes)
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Precisión-Recall curve de la clase {0} (AP = {1:0.2f})'
                 ''.format(classes[i], average_precision[i]))
    
    plt.xlabel('Recall')
    plt.ylabel('Precisión')
    plt.title('Curvas de Precisión-Recall para Multiclasificación')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    default_logger.info(f"Curvas Precisión-Recall guardadas en: {save_path}")

def plot_multilabel_confusion_matrix(y_true, y_pred, labels, save_path):
    """
    Genera y guarda matrices de confusión para cada clase en un modelo multietiqueta.
    """
    # Binarizar las etiquetas
    y_true_binarized = y_true
    y_pred_binarized = y_pred
    
    n_classes = y_true_binarized.shape[1]
    
    for i in range(n_classes):
        cm = confusion_matrix(y_true_binarized[:, i], y_pred_binarized[:, i])
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        plt.title(f"Matriz de Confusión - {labels[i]}")
        plt.tight_layout()
        cm_save_path = os.path.join(save_path, f'confusion_matrix_{labels[i]}.png')
        plt.savefig(cm_save_path)
        plt.close()
        default_logger.info(f"Matriz de confusión para '{labels[i]}' guardada en: {cm_save_path}")

def plot_multilabel_roc(y_true, y_pred_proba, classes, save_path):
    """
    Genera y guarda las curvas ROC para cada clase en un modelo multietiqueta.
    """
    n_classes = y_pred_proba.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10,8))
    colors = sns.color_palette("bright", n_classes)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve de la clase {0} (AUC = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC para Multietiquetas')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    default_logger.info(f"Curvas ROC multietiqueta guardadas en: {save_path}")

def plot_multilabel_precision_recall(y_true, y_pred, classes, save_path):
    """
    Genera y guarda las curvas de Precisión-Recall para cada clase en un modelo multietiqueta.
    """

    y_true_binarized = y_true
    y_pred_binarized = y_pred
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
        average_precision[i] = average_precision_score(y_true_binarized[:, i], y_pred_binarized[:, i])
    
    plt.figure(figsize=(10,8))
    colors = sns.color_palette("bright", len(classes))
    for i, color in zip(range(len(classes)), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Precisión-Recall curve de la clase {0} (AP = {1:0.2f})'
                 ''.format(classes[i], average_precision[i]))
    
    plt.xlabel('Recall')
    plt.ylabel('Precisión')
    plt.title('Curvas de Precisión-Recall para Multietiquetas')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    default_logger.info(f"Curvas Precisión-Recall multietiqueta guardadas en: {save_path}")

def display_metrics(metrics, graph_subdir):
    """
    Muestra las métricas del modelo entrenado y genera gráficas.

    Args:
        metrics (dict): Diccionario con las métricas del modelo.
        graph_subdir (str): Subdirectorio dentro de 'static/graphs' donde se guardarán las gráficas.
    """
    print("\nMétricas del Modelo Entrenado:")
    print("="*80)
    print(f"Precisión del modelo: {metrics['accuracy'] * 100:.2f}%")
    print("Matriz de Confusión:")
    print(metrics['confusion_matrix'])
    print("\nReporte de Clasificación:")
    print(metrics['classification_report'])
    print("-"*80)
    
    graphs_dir = create_graphs_subdir(graph_subdir)
    
    # Generar y guardar la matriz de confusión
    cm_title = "Matriz de Confusión - Modelo de Multiclasificación"
    cm_save_path = os.path.join(graphs_dir, 'confusion_matrix_multiclass.png')
    plot_confusion_matrix_custom(
        y_true=metrics['y_test'],
        y_pred=metrics['y_pred'],
        labels=['circle', 'square', 'triangle', 'star'],
        title=cm_title,
        save_path=cm_save_path
    )
    
    # Generar y guardar el reporte de clasificación
    cr_title = "Reporte de Clasificación - Modelo de Multiclasificación"
    cr_save_path = os.path.join(graphs_dir, 'classification_report_multiclass.png')
    plot_classification_report_custom(
        y_true=metrics['y_test'],
        y_pred=metrics['y_pred'],
        labels=['circle', 'square', 'triangle', 'star'],
        title=cr_title,
        save_path=cr_save_path
    )
    
    # Generar y guardar las curvas ROC
    if hasattr(metrics['model'], 'predict_proba'):
        y_pred_proba = metrics['model'].predict_proba(metrics['X_test'])
        roc_save_path = os.path.join(graphs_dir, 'roc_multiclass.png')
        plot_multiclass_roc(
            y_true=metrics['y_test'],
            y_pred_proba=y_pred_proba,
            classes=['circle', 'square', 'triangle', 'star'],
            save_path=roc_save_path
        )
    
        # Generar y guardar las curvas Precision-Recall
        pr_save_path = os.path.join(graphs_dir, 'precision_recall_multiclass.png')
        plot_multiclass_precision_recall(
            y_true=metrics['y_test'],
            y_pred_proba=y_pred_proba,
            classes=['circle', 'square', 'triangle', 'star'],
            save_path=pr_save_path
        )

def display_metrics_multilabel(model, y_test, y_pred, X_test, label_names, graph_subdir):
    """
    Muestra las métricas del modelo multietiqueta y genera gráficas.

    Args:
        model: Modelo multietiqueta entrenado.
        y_test (ndarray): Etiquetas verdaderas.
        y_pred (ndarray): Etiquetas predichas.
        X_test (ndarray): Características de prueba.
        label_names (list): Lista de nombres de etiquetas.
        graph_subdir (str): Subdirectorio dentro de 'static/graphs' donde se guardarán las gráficas.
    """
    print("\nMétricas del Modelo Multietiqueta:")
    print("="*80)
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))
    print(f"Hamming Loss: {hamming_loss(y_test, y_pred):.4f}")
    print(f"Jaccard Score (macro): {jaccard_score(y_test, y_pred, average='macro'):.4f}")
    print("-"*80)
    
    # Crear directorio para gráficas específicas
    graphs_dir = create_graphs_subdir(graph_subdir)
    
    # Generar y guardar el reporte de clasificación multietiqueta
    cr_title = "Reporte de Clasificación - Modelo Multietiqueta"
    cr_save_path = os.path.join(graphs_dir, 'classification_report_multilabel.png')
    plot_classification_report_custom(
        y_true=y_test,
        y_pred=y_pred,
        labels=label_names,
        title=cr_title,
        save_path=cr_save_path
    )
    
    # Generar y guardar las curvas Precision-Recall
    pr_save_path = os.path.join(graphs_dir, 'precision_recall_multilabel.png')
    plot_multilabel_precision_recall(
        y_true=y_test,
        y_pred=y_pred,
        classes=label_names,
        save_path=pr_save_path
    )
    
    # Generar y guardar las matrices de confusión multietiqueta
    plot_multilabel_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        labels=label_names,
        save_path=graphs_dir
    )
    
    # Generar y guardar las curvas ROC multietiqueta si el modelo las soporta
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
        roc_save_path = os.path.join(graphs_dir, 'roc_multilabel.png')
        plot_multilabel_roc(
            y_true=y_test,
            y_pred_proba=y_pred_proba,
            classes=label_names,
            save_path=roc_save_path
        )

def main():
    # Definir las figuras disponibles y su mapeo de etiquetas
    label_names = ['circle', 'square', 'triangle', 'star']
    label_mapping = {name: idx for idx, name in enumerate(label_names)}

    # Directorios
    shapes_dir = os.path.join(os.getcwd(), 'static', 'shapes')
    preprocessing_dir = os.path.join(os.getcwd(), 'static', 'preprocessing_shapes')
    combined_images_dir = os.path.join(shapes_dir, 'combined')
    labels_file = os.path.join(combined_images_dir, 'labels.csv')
    
    class_dir = os.path.join(os.getcwd(), 'static', 'class')
    multilabel_dir = os.path.join(os.getcwd(), 'static', 'multilabel')
    
    annotated_dir = os.path.join(os.getcwd(), 'static', 'annotated')

    os.makedirs(preprocessing_dir, exist_ok=True)
    os.makedirs(combined_images_dir, exist_ok=True)
    os.makedirs(class_dir, exist_ok=True)          
    os.makedirs(multilabel_dir, exist_ok=True)     
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(os.path.join('static', 'graphs', 'class'), exist_ok=True) 
    os.makedirs(os.path.join('static', 'graphs', 'multilabel'), exist_ok=True)

    # Menú de opciones
    print("Selecciona una opción para el flujo de trabajo:")
    print("P. Preprocesar imágenes y generar dataset")
    print("A. Entrenar modelo de multiclasificación")
    print("B. Entrenar modelo multietiqueta")
    print("1. Clasificar imágenes con una sola figura")
    print("2. Detectar y clasificar múltiples figuras en imágenes")
    choice = input("Ingresa P, A, B, 1 o 2: ").strip().upper()

    if choice == 'P':
        # Preprocesar imágenes y generar dataset
        preprocess_images(shapes_dir, preprocessing_dir, test_ratio=0.1)

        try:
            subprocess.run([sys.executable, '-m', 'core.combine_images'], check=True)
            print("combine_images.py se ejecutó correctamente.")
            default_logger.info("combine_images.py se ejecutó correctamente.")
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar combine_images.py: {e}")
            default_logger.error(f"Error al ejecutar combine_images.py: {e}")

    elif choice == 'A':
        # Entrenar modelo de multiclasificación
        X_train, y_train, X_test, y_test = load_dataset(preprocessing_dir)

        if X_train.size == 0 or X_test.size == 0:
            print("No hay suficientes datos para entrenar o evaluar el modelo.")
            return

        # Entrenar el modelo
        model, metrics = train_model(X_train, y_train, X_test, y_test)
        
        # Añadir información adicional al diccionario de métricas
        metrics['y_test'] = y_test
        metrics['y_pred'] = model.predict(X_test)
        metrics['model'] = model
        metrics['X_test'] = X_test
        
        save_model(model, model_name='model.pkl')
        display_metrics(metrics, graph_subdir='class')  # Guardar en 'graphs/class'

    elif choice == 'B':
        # Opción B: Entrenar modelo multietiqueta
        print("\nEntrenando modelo multietiqueta...")
        default_logger.info("Entrenando modelo multietiqueta...")

        # Asegúrate de que los directorios son correctos
        combined_images_dir = os.path.join(shapes_dir, 'combined')
        labels_file = os.path.join(combined_images_dir, 'labels.csv')

        # Cargar el dataset multietiqueta
        X_train, y_train, X_test, y_test = load_multilabel_dataset(combined_images_dir, labels_file)

        if X_train.size == 0 or X_test.size == 0:
            print("No hay suficientes datos para entrenar o evaluar el modelo.")
            return

        # Entrenar el modelo multietiqueta
        model = train_multilabel_model(X_train, y_train)
        save_model(model, model_name='multilabel_model.pkl')

        # Evaluar el modelo
        y_pred = model.predict(X_test)
        display_metrics_multilabel(model, y_test, y_pred, X_test, label_names, graph_subdir='multilabel')  # Guardar en 'graphs/multilabel'

    elif choice == '1':
        # Clasificar imágenes con una sola figura
        model = load_model(model_name='model.pkl')
        if not model:
            print("No se encontró un modelo entrenado. Por favor, entrena el modelo primero (Opción A).")
            return

        image_files = [img for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print(f"No se encontraron imágenes en {class_dir} para la predicción.")
            return

        annotated_images = []

        for image_name in image_files:
            image_path = os.path.join(class_dir, image_name)
            features = preprocess_image_for_classification(image_path)
            if features is None:
                continue

            features = features.reshape(1, -1)
            prediction = model.predict(features)[0]
            shape_name = label_names[prediction]

            print(f"Imagen: {image_name} - Figura Detectada: {shape_name}")
            default_logger.info(f"Imagen: {image_name} - Figura Detectada: {shape_name}")

            # Anotar la imagen
            img = cv2.imread(image_path)
            if img is not None:
                label_text = f"Figura: {shape_name}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
                text_x = 10
                text_y = 20
                overlay = img.copy()
                cv2.rectangle(overlay, (text_x - 5, text_y - text_size[1] - 5), 
                              (text_x + text_size[0] + 5, text_y + 5), 
                              (0, 0, 0), -1)
                alpha = 0.6
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                cv2.putText(img, label_text, (text_x, text_y),
                            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
                annotated_image_path = os.path.join(annotated_dir, f"annotated_{image_name}")
                cv2.imwrite(annotated_image_path, img)
                print(f"Imagen anotada guardada en: {annotated_image_path}")

                annotated_images.append((img, image_name, shape_name))

        # Visualizar las imágenes anotadas en una figura
        if annotated_images:
            num_images = len(annotated_images)
            cols = 3
            rows = num_images // cols + int(num_images % cols > 0)
            plt.figure(figsize=(15, 5 * rows))
            for idx, (img, image_name, shape_detected) in enumerate(annotated_images):
                plt.subplot(rows, cols, idx + 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img_rgb)
                plt.axis('off')
                plt.title(f"{image_name}\nFigura: {shape_detected}", fontsize=10)
            plt.suptitle('Imágenes Anotadas - Clasificación de una Sola Figura', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            annotated_plot_path = os.path.join('static', 'graphs', 'class', 'annotated_images_plot_option1.png')
            plt.savefig(annotated_plot_path)
            plt.close()
            default_logger.info(f"Figura de imágenes anotadas guardada en: {annotated_plot_path}")
            print(f"Figura de imágenes anotadas guardada en: {annotated_plot_path}")

    elif choice == '2':
        # Detectar y clasificar múltiples figuras en imágenes (usando modelo multietiqueta)
        model = load_model(model_name='multilabel_model.pkl')
        if not model:
            print("No se encontró un modelo multietiqueta entrenado. Por favor, entrena el modelo primero (Opción B).")
            return

        image_files = [img for img in os.listdir(multilabel_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print(f"No se encontraron imágenes en {multilabel_dir} para la predicción.")
            return

        annotated_images = []

        for image_name in image_files:
            image_path = os.path.join(multilabel_dir, image_name)
            features = preprocess_image_for_classification(image_path)
            if features is None:
                continue

            features = features.reshape(1, -1)
            prediction = model.predict(features)[0]

            if isinstance(prediction, (list, np.ndarray)):
                prediction = np.array(prediction)
            else:
                prediction = np.array([prediction])
            predicted_labels = [label_names[i] for i, val in enumerate(prediction) if val == 1]

            print(f"Imagen: {image_name} - Figuras Detectadas: {', '.join(predicted_labels)}")
            default_logger.info(f"Imagen: {image_name} - Figuras Detectadas: {', '.join(predicted_labels)}")

            # Anotar la imagen
            img = cv2.imread(image_path)
            if img is not None:
                label_text = "Figuras: " + ", ".join(predicted_labels)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5 
                thickness = 1  
                text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
                text_x = 10
                text_y = 20  

                overlay = img.copy()
                cv2.rectangle(overlay, (text_x - 5, text_y - text_size[1] - 5), 
                              (text_x + text_size[0] + 5, text_y + 5), 
                              (0, 0, 0), -1)
                alpha = 0.6  
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                # Escribir el texto
                cv2.putText(img, label_text, (text_x, text_y),
                            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
                annotated_image_path = os.path.join(annotated_dir, f"annotated_{image_name}")
                cv2.imwrite(annotated_image_path, img)
                print(f"Imagen anotada guardada en: {annotated_image_path}")

                annotated_images.append((img, image_name, predicted_labels))

        # Visualizar las imágenes anotadas en una figura
        if annotated_images:
            num_images = len(annotated_images)
            cols = 3
            rows = num_images // cols + int(num_images % cols > 0)
            plt.figure(figsize=(15, 5 * rows))
            for idx, (img, image_name, labels_detected) in enumerate(annotated_images):
                plt.subplot(rows, cols, idx + 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img_rgb)
                plt.axis('off')
                plt.title(f"{image_name}\nFiguras: {', '.join(labels_detected)}", fontsize=10)
            plt.suptitle('Imágenes Anotadas - Detección y Clasificación de Múltiples Figuras', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            annotated_plot_path = os.path.join('static', 'graphs', 'multilabel', 'annotated_images_plot_option2.png')
            plt.savefig(annotated_plot_path)
            plt.close()
            default_logger.info(f"Figura de imágenes anotadas guardada en: {annotated_plot_path}")
            print(f"Figura de imágenes anotadas guardada en: {annotated_plot_path}")

    else:
        print("Opción no válida seleccionada.")
        return

if __name__ == "__main__":
    main()