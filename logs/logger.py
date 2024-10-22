import logging
import os

def setup_logger(name='default', log_file='app.log', level=logging.INFO):
    """
    Configura y retorna un logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Crear directorio de logging si no existe
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, log_file)
    
    # Crear manejador de archivo
    handler = logging.FileHandler(log_path)
    handler.setLevel(level)
    
    # Crear formato
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Añadir manejador al logger
    if not logger.hasHandlers():
        logger.addHandler(handler)
    
    return logger

# Logger por defecto
default_logger = setup_logger()