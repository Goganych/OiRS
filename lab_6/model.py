from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class Autoencoder:
	# Размерность кодированного представления
	encoding_dim = 128

	def build(self, height, width, channels):
		# Энкодер
		# вход encoder
		input_img = Input(shape=(height, width, channels))

		# "вытянуть" картинку в одномерный вектор
		flat_img = Flatten()(input_img)
		# Кодированное полносвязным слоем представление
		x = Dense(self.encoding_dim*4, activation='relu')(flat_img)
		x = Dense(self.encoding_dim*3, activation='relu')(x)
		x = Dense(self.encoding_dim*2, activation='relu')(x)
		encoded = Dense(self.encoding_dim, activation='linear')(x)
		
		# Декодер
		# Раскодированное другим полносвязным слоем изображение
		input_encoded = Input(shape=(self.encoding_dim,))
		x = Dense(self.encoding_dim*2, activation='relu')(input_encoded)
		x = Dense(self.encoding_dim*3, activation='relu')(x)
		x = Dense(self.encoding_dim*4, activation='relu')(x)
		flat_decoded = Dense(height*width*channels, activation='linear')(x)

		decoded = Reshape((height, width, channels))(flat_decoded)

		# Модели, в конструктор первым аргументом передаются входные слои, а вторым выходные слои
		# Другие модели можно так же использовать как и слои
		encoder = Model(input_img, encoded, name="encoder")
		decoder = Model(input_encoded, decoded, name="decoder")

		autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
		return autoencoder
