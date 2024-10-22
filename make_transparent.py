from PIL import Image

def make_background_transparent(shape_image_path, output_path, threshold=240):
    """
    Convierte el fondo blanco de una imagen a transparente.
    
    :param shape_image_path: Ruta de la imagen original con fondo blanco.
    :param output_path: Ruta donde se guardará la imagen con fondo transparente.
    :param threshold: Umbral para considerar un píxel como blanco.
    """
    image = Image.open(shape_image_path).convert("RGBA")
    datas = image.getdata()

    newData = []
    for item in datas:
        # Si el color es blanco (o cercano), hacerlo transparente
        if item[0] > threshold and item[1] > threshold and item[2] > threshold:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    image.putdata(newData)
    image.save(output_path, "PNG")

if __name__ == "__main__":
    import os

    shapes_dir = os.path.join(os.getcwd(), 'static', 'shapes')
    for shape in ['circle', 'square', 'triangle', 'star']:
        shape_folder = os.path.join(shapes_dir, shape)
        for image_name in os.listdir(shape_folder):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(shape_folder, image_name)
                output_path = os.path.join(shape_folder, f"trans_{image_name.split('.')[0]}.png")
                make_background_transparent(image_path, output_path)
                print(f"Procesada: {output_path}")