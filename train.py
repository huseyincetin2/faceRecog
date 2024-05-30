from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from sklearn.metrics import classification_report
from keras.layers import Dropout
import matplotlib.pyplot as plt


# CNN modelinin oluşturulması
classifier = Sequential()

# Adım 1 - Filtreleme (Convolution)
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# Adım 2 - Örnekleme (Pooling)
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#classifier.add(Dropout(0.25))

# İkinci Evrişim Katmanı
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#classifier.add(Dropout(0.25))

# üçüncü Evrişim Katmanı
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# üçüncü Evrişim Katmanı
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adım 3 - Düzleştirme (Flattening)
classifier.add(Flatten())

# Adım 4 - Tam Bağlantı (Full Connection)
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=32, activation='softmax'))  # 105 sınıf için softmax kullanıyoruz

# CNN modelinin derlenmesi
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Veri Artırımı ve Ön İşleme
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim ve test veri kümelerinin hazırlanması ve ön işlenmesi
training_set = train_datagen.flow_from_directory('./dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='categorical', subset='training')
validation_set = train_datagen.flow_from_directory('./dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='categorical', subset='validation')
test_set = test_datagen.flow_from_directory('./dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='categorical')

# Erken durdurma geri çağrısını tanımlama
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# CNN modelinin eğitimi
history = classifier.fit(training_set,
    steps_per_epoch=len(training_set),
    epochs=80,
    validation_data=validation_set,
    validation_steps=len(validation_set),
    validation_split=0.2,
    callbacks=[early_stopping]
)


# Test veri setinden tahminlerin alınması
predictions = classifier.predict(test_set)
y_pred = np.argmax(predictions, axis=1)

# Gerçek etiketlerin alınması
y_true = test_set.classes

# Classification raporunun alınması
report = classification_report(y_true, y_pred)

#
print(report)

# Modelin kaydedilmesi
classifier.save('./model/m9.keras')


classifier.summary()


plt.figure()
plt.title("Model Accuracy")
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()
plt.show()

plt.figure()
plt.title("Model Loss")
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.legend()
plt.show()
