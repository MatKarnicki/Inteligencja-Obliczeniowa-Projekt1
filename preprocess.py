from PIL import Image
import os
import cv2
import torchvision
from torchvision.transforms import transforms
def load_photos_as_tensors(directory_path):
    photo_tensors = []
    i = 0
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    processed_dir = 'processed2'
    os.makedirs(processed_dir, exist_ok=True)
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(directory_path, filename))
            img_pil = Image.fromarray(img)
            img_transformed = transform(img_pil)
            photo_tensors.append(img_transformed)
            save_path = os.path.join(processed_dir, f'sori_processed_{i}.png')
            torchvision.utils.save_image(img_transformed, save_path)
            i += 1


load_photos_as_tensors("preprocessed/")
