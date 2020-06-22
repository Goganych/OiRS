from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
import numpy as np

class Autoencoder:
	# Размерность кодированного представления
	encoding_dim = 256

	def build(self, height, width, channels):
		# Энкодер
		# вход encoder
		filters=(128, 64)
		input_img = Input(shape=(height, width, channels))
		x=input_img
		# "вытянуть" картинку в одномерный вектор
		for f in filters:
			# apply a CONV => RELU => BN operation
			x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=-1)(x)
		volumeSize = K.int_shape(x)
		flat_img = Flatten()(x)
		# Кодированное полносвязным слоем представление
		encoded = Dense(self.encoding_dim, activation='relu')(flat_img)
		
		# Декодер
		# Раскодированное другим полносвязным слоем изображение
		input_encoded = Input(shape=(self.encoding_dim))
		x = Dense(np.prod(volumeSize[1:]))(input_encoded)
		x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
		for f in filters[::-1]:
			# apply a CONV_TRANSPOSE => RELU => BN operation
			x = Conv2DTranspose(f, (3, 3), strides=2,
				padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=-1)(x)
		x = Conv2DTranspose(3, (3, 3), padding="same")(x)
		decoded = Activation("sigmoid")(x)

		# Модели, в конструктор первым аргументом передаются входные слои, а вторым выходные слои
		# Другие модели можно так же использовать как и слои
		encoder = Model(input_img, encoded, name="encoder")
		decoder = Model(input_encoded, decoded, name="decoder")

		autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
		return autoencoder
