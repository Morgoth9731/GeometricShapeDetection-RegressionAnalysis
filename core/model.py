from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os
from logs.logger import default_logger

def train_model(X, y):
    """
    Entrena un modelo de regresión logística y retorna el modelo entrenado.
    
    :param X: Características de entrenamiento.
    :param y: Etiquetas de entrenamiento.
    :return: Modelo entrenado.
    """
    # Dividimos los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos el modelo de regresión logística con balanceo de clases
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)  # Entrenamos el modelo

    # Hacemos predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calculamos la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy * 100:.2f}%")
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # Logs
    default_logger.info(f"Precisión del modelo: {accuracy * 100:.2f}%")
    default_logger.info(f"Matriz de Confusión:\n{confusion_matrix(y_test, y_pred)}")
    default_logger.info(f"Reporte de Clasificación:\n{classification_report(y_test, y_pred)}")

    return model

def save_model(model, figure):
    """
    Guarda el modelo entrenado en la carpeta 'models'.
    
    :param model: Modelo entrenado.
    :param figure: Nombre de la figura.
    """
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_filename = os.path.join(models_dir, f'{figure}_model.pkl')
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    default_logger.info(f"Modelo para '{figure}' guardado como {model_filename}")
    print(f"Modelo para '{figure}' guardado como {model_filename}")

def load_model(figure):
    """
    Carga un modelo entrenado desde la carpeta 'models'.
    
    :param figure: Nombre de la figura.
    :return: Modelo cargado o None si no se encuentra.
    """
    model_filename = os.path.join(os.getcwd(), 'models', f'{figure}_model.pkl')
    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
        default_logger.info(f"Modelo para '{figure}' cargado desde {model_filename}")
        print(f"Modelo para '{figure}' cargado desde {model_filename}")
        return model
    else:
        default_logger.warning(f"No se encontró el modelo para '{figure}' en {model_filename}")
        print(f"No se encontró el modelo para '{figure}' en {model_filename}")
        return None