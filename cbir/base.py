def resizeIMG(img_path):
    from PIL import Image
    image = Image.open(img_path)
    sunset_resized = image.resize((512, 512))
    sunset_resized.save(img_path)
    return