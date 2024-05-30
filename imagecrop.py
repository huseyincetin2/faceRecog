import cv2
import os
import dlib

# Yüz dedektörü
detector = dlib.get_frontal_face_detector()

# Görüntü klasörünün yolu
image_folder = "cropImage/"

# Output klasörünün yolu
output_folder = "output/"

# Output klasöründe kaydedilen dosyaların listesi
existing_faces = [file for file in os.listdir(output_folder) if file.startswith("face")]

# Output klasöründe kaydedilen en yüksek numaralı dosyanın bulunması
if existing_faces:
    last_face_num = max([int(file.split("_")[1].split(".")[0]) for file in existing_faces])
else:
    last_face_num = 0


# Yüzleri crop edip kaydetme fonksiyonu
def crop_faces(image_path):
    print(image_path)
    global last_face_num
    # Görüntüyü oku
    img = cv2.imread(image_path)
    # Gri tonlamalı yap
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Yüzleri tespit et
    faces = detector(gray)

    # Eğer yüz yoksa fonksiyonu bitir 
    if not faces:
        return

    # Yüzleri crop et ve kaydet
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        # Yüzü crop et
        face_img = img[y:y + h, x:x + w]
        # Yeni dosya adını belirle
        new_file_name = f"face_{last_face_num + 1}.jpg"
        while new_file_name in existing_faces:
            last_face_num += 1
            new_file_name = f"face_{last_face_num + 1}.jpg"
        # Yüzü dosyaya kaydet
        cv2.imwrite(os.path.join(output_folder, new_file_name), face_img)
        # Dosya listesini güncelle
        existing_faces.append(new_file_name)


# Görüntü klasöründeki her bir dosya için yüzleri tespit et ve crop et
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        crop_faces(image_path)
