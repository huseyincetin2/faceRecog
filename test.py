from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Eğitilmiş modeli yükleme
model_path = 'model/m8.keras'  # Eğittiğiniz modelin dosya yolu
model = load_model(model_path, compile=False)  # compile=False parametresiyle modeli optimizer'ı oluşturmadan yükleyin

# Tahmin edilecek resmin dosya yolu
test_image_path = './testImages/img_7.png'  # Test edilecek resmin dosya yolu

# Resmi model için uygun formata getirme
test_image = image.load_img(test_image_path, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image /= 255.  # Resmi normalize etme

# Tahmin yapma
result = model.predict(test_image)
# print(result)
predicted_class_val = np.max(result)
# print(predicted_class_val)
predicted_class_index = np.argmax(result)
# print(predicted_class_index)

folder_path = './dataset/test_set'
files = os.listdir(folder_path)
class_labels = sorted(files)
# print(class_labels)

predicted_label = class_labels[predicted_class_index]
print("Belirtilen Görseldeki Kişi:", predicted_label)
