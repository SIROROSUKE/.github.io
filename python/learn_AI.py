import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import keras
from matplotlib.font_manager import FontProperties
import scipy

epochs = 200
# 新しい画像のパスを指定します。
img_path = "C:/Users/owner/Desktop/various/python/AI_program/medaka-AI/image_test/o0800044713441853173.jpg"
#'C:/Users/owner/Desktop/various/python/AI_program/medaka-AI/image_folder/1_img/main2_img.jpg'
# 画像データのディレクトリを指定します。
data_dir = 'C:/Users/owner/Desktop/various/python/AI_program/medaka-AI/image_folder'
plt.font_manager.FontProperties(fname = 'C:\WINDOWS\Fonts\msgothic.ttc' ,size = 15 )

# 画像データの前処理を行います。
image_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    subset='training')

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    subset='validation')

# モデルの構築を行います。
model = keras.Sequential([
    keras.layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((3, 3)),
    keras.layers.Dropout(0.15),
    keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((3, 3)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l1(0.01), input_shape=(1,)),
    keras.layers.Dense(train_gen.num_classes, activation='softmax')
])

# モデルの重みを作成します。
model.build(input_shape=(None, 224, 224, 3))

#conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu')
# モデルのコンパイルを行います。
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# モデルの学習を行います。
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen)

# モデル全体をKerasネイティブ形式ファイルに保存します。
model.save('C:/Users/owner/Desktop/web_try/learning_data/200_learning_data')

# 画像を読み込み、前処理を行います。
img = image.load_img(img_path, target_size=image_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# モデルを使用して画像を分類します。
predictions = model.predict(x)
predicted_class = np.argmax(predictions[0])
class_names = train_gen.class_indices.keys()
class_names = list(train_gen.class_indices.keys())
predictions = model.predict(x)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]


result = "Non"
if class_names[predicted_class] == "1_img":
    result = "黒メダカ"
if class_names[predicted_class] == "2_img":
    result = "ヒメダカ"
if class_names[predicted_class] == "3_img":
    result = "ハヤ（ウグイ）の稚魚"
if class_names[predicted_class] == "4_img":
    result = "楊貴妃メダカ"
if class_names[predicted_class] == "5_img":
    result = "ラメメダカ"

# 精度のグラフをプロットします。
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('モデルの精度',fontname = 'MS Gothic')
plt.ylabel('精度',fontname = 'MS Gothic')
plt.xlabel('学習回数',fontname = 'MS Gothic')
plt.legend(['学習用モデル', '評価モデル'], loc='upper left',prop={"family":"MS Gothic"})
#plt.show()

# 損失のグラフをプロットします。
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('モデルの損失',fontname = 'MS Gothic')
plt.ylabel('損失',fontname = 'MS Gothic')
plt.xlabel('学習回数',fontname = 'MS Gothic')
plt.legend(['学習用モデル', '評価モデル'], loc='upper left',prop={"family":"MS Gothic"})
plt.show()

print('予測結果:', result,"  ,  信頼性：",confidence * 100,"%")