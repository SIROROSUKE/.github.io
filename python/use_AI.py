import keras
import numpy as np
from keras.preprocessing import image

#配列の定義
names = ["black Rice fish","Scarlet Rice fish","ハヤの稚魚","楊貴妃メダカ","ラメメダカ"]

# モデル全体をSavedModel形式のファイルから読み込みます。
model = keras.models.load_model('C:/Users/owner/Desktop/web_try/learning_data/1000_learning_data')

# 画像のパスを指定します。
img_path = 'C:/Users/owner/Desktop/various/python/AI_program/medaka-AI/image_test/IMG_2133.jpeg'

# 画像サイズを指定します。
image_size = (224, 224)

# 画像を読み込み、前処理を行います。
img = image.load_img(img_path, target_size=image_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# モデルを使用して画像を分類します。
predictions = model.predict(x)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

reslt = names[predicted_class]

# 分類結果と信頼度を表示します。
print('Predicted class:', reslt)
print('Confidence:', confidence * 100)