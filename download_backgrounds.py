import os
import requests
from tqdm import tqdm
import sys

def download_image(url, save_path):
    """
    Descarga una imagen desde una URL y la guarda en la ruta especificada.
    
    :param url: URL de la imagen a descargar.
    :param save_path: Ruta donde se guardará la imagen.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Verifica que la solicitud fue exitosa
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err} - {url}')
    except Exception as err:
        print(f'Other error occurred: {err} - {url}')

def get_random_images(access_key, count=10, orientation='landscape'):
    """
    Obtiene URLs de imágenes aleatorias desde la API de Unsplash.
    
    :param access_key: Tu clave de acceso a la API de Unsplash.
    :param count: Número de imágenes a obtener (máximo 30 por solicitud).
    :param orientation: Orientación de las imágenes ('landscape', 'portrait', 'squarish').
    :return: Lista de URLs de imágenes.
    """
    url = 'https://api.unsplash.com/photos/random'
    headers = {
        'Accept-Version': 'v1',
        'Authorization': f'Client-ID {access_key}'
    }
    params = {
        'count': count,
        'orientation': orientation
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        image_urls = [img['urls']['regular'] for img in data]
        return image_urls
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return []
    except Exception as err:
        print(f'Other error occurred: {err}')
        return []

def main():
    # Configuración
    access_key = os.getenv('UNSPLASH_ACCESS_KEY')  # Obtener la API key desde una variable de entorno
    if not access_key:
        print("Error: Por favor, establece la variable de entorno 'UNSPLASH_ACCESS_KEY' con tu Access Key de Unsplash.")
        sys.exit(1)
    
    backgrounds_dir = os.path.join(os.getcwd(), 'static', 'backgrounds')
    os.makedirs(backgrounds_dir, exist_ok=True)
    
    # Número de imágenes a descargar
    try:
        num_images = int(input("¿Cuántas imágenes de fondo deseas descargar? "))
    except ValueError:
        print("Entrada inválida. Por favor, ingresa un número entero.")
        sys.exit(1)
    
    # Orientación de las imágenes
    orientation = input("Selecciona la orientación de las imágenes ('landscape', 'portrait', 'squarish'): ").strip().lower()
    if orientation not in ['landscape', 'portrait', 'squarish']:
        print("Orientación no válida. Usando 'landscape' por defecto.")
        orientation = 'landscape'
    
    # Descargar imágenes
    batch_size = 30  # Máximo permitido por solicitud de la API
    images_downloaded = 0
    with tqdm(total=num_images, desc="Descargando imágenes") as pbar:
        while images_downloaded < num_images:
            remaining = num_images - images_downloaded
            current_batch = min(batch_size, remaining)
            image_urls = get_random_images(access_key, count=current_batch, orientation=orientation)
            if not image_urls:
                print("No se pudieron obtener más imágenes. Terminando la descarga.")
                break
            for url in image_urls:
                image_extension = os.path.splitext(url)[1].split('?')[0]  # Obtener la extensión
                if image_extension.lower() not in ['.jpg', '.jpeg', '.png']:
                    image_extension = '.jpg'  # Por defecto
                image_name = f'background_{images_downloaded + 1}{image_extension}'
                save_path = os.path.join(backgrounds_dir, image_name)
                download_image(url, save_path)
                images_downloaded += 1
                pbar.update(1)
                if images_downloaded >= num_images:
                    break
    print(f"Descarga completada: {images_downloaded} imágenes guardadas en {backgrounds_dir}")

if __name__ == "__main__":
    main()