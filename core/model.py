import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from logs.logger import default_logger
from sklearn.metrics import classification_report, confusion_matrix

def train_model(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo de Regresión Logística para multiclasificación.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    default_logger.info("Modelo de multiclasificación entrenado exitosamente.")

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }

    return model, metrics

def train_multilabel_model(X_train, y_train):
    """
    Entrena un modelo de Regresión Logística para clasificación multietiqueta.
    """
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    default_logger.info("Modelo multietiqueta entrenado exitosamente.")

    return model

def save_model(model, model_name='model.pkl'):
    """
    Guarda el modelo entrenado.
    """
    models_dir = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_path = os.path.join(models_dir, model_name)
    joblib.dump(model, model_path)
    default_logger.info(f"Modelo guardado en: {model_path}")

def load_model(model_name='model.pkl'):
    """
    Carga el modelo entrenado.
    """
    model_path = os.path.join(os.getcwd(), 'models', model_name)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        default_logger.info(f"Modelo cargado desde: {model_path}")
        return model
    else:
        default_logger.error(f"No se encontró el modelo en {model_path}")
        return None