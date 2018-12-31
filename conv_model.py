from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def predict(val, blackAndWhite):
	first_img = orig_stereos[val]
	test_plot_image(first_img, blackAndWhite)
	output = stereo_model.predict(np.array([stereos[val]]), batch_size = 1)
	test_plot_image(orig_depths[val], blackAndWhite)
	test_plot_image(np.reshape(output[0], [100,150]), blackAndWhite)

# This is to load an image from pillow and convert it into a numpy array
def load_image(infilename, blackAndWhite):
	img = Image.open(infilename)
	if blackAndWhite:
		img = img.convert("L")
	img.load()
	data = np.asarray( img, dtype="int32" )
	return data

# This is to display the image
def test_plot_image(input, blackAndWhite):
	if blackAndWhite:
		plt.imshow(input, cmap = "Greys")
	else:
		plt.imshow(input)
	plt.show()

# hyperparameters
shape_of_input_color = (100, 150, 8)
shape_of_output_color = (100, 150, 4)
shape_of_input = (100, 150, 2)
shaoe_of_output = (100, 150, 1)

# This is to import and preprocess the data
def import_data(blackAndWhite):
	stereos = sorted(glob.glob("./StereoImages/*.png"))
	depths = sorted(glob.glob("./Depth_map/*.png"))
	stereos = np.array([load_image(stereo, blackAndWhite) for stereo in stereos])
	depths = np.array([load_image(depth, blackAndWhite) for depth in depths])
	if blackAndWhite:
		stereos_after = np.array([np.reshape(i, [100,300,1]) for i in stereos])
		depths_after = np.array([np.reshape(i, [100,150,1]) for i in depths])
	else:
		stereos_after = stereos
		depths_after = depths

	new_stereos = []
	for i in range(stereos_after.shape[0]):
		test_image = stereos_after[i]
		image1 = test_image[:,:150,:]
		image2 = test_image[:,150:,:]
		new_image = np.concatenate([image1, image2], axis = 2)
		new_stereos.append(new_image)

	new_stereos = np.array(new_stereos)

	steroes_offset = np.mean(new_stereos, 0)
	steroes_scale = np.std(new_stereos, 0).clip(min=1)
	depths_scale = depths.max()

	steroes_old_offset = np.mean(stereos, 0)
	steroes_old_scale = np.std(stereos, 0).clip(min = 1)

	firstOutput = (new_stereos - steroes_offset)/steroes_scale
	secondOutput = (depths_after)/depths_scale
	thirdOutput = (stereos - steroes_old_offset)/(steroes_old_scale)
	lastOutput = depths/(depths_scale)

	return firstOutput, secondOutput, thirdOutput, lastOutput 

# This is the model creation
def createModel(blackAndWhite):
	def createConvLayer(num_filters, input_tensor):
		conv = SeparableConv2D(num_filters, (3,3), data_format = "channels_last", padding = "same")(input_tensor)
		conv = BatchNormalization()(conv)
		conv = Activation("relu")(conv)
		return conv

	if blackAndWhite:
		visible = Input(shape_of_input)
	else:
		visible = Input(shape_of_input_color)
	conv1 = createConvLayer(32, visible)
	maxpool1 = MaxPooling2D((2,2))(conv1)
	conv2 = createConvLayer(64, maxpool1)
	conv3 = createConvLayer(64, conv2)
	upscale = UpSampling2D((2,2))(conv3)
	conv4 = createConvLayer(32, upscale)
	if blackAndWhite:
		output = createConvLayer(1, conv4)
	else:
		output = createConvLayer(4, conv4)
	model = Model(inputs = visible, outputs = output)
	return model

def createNewModel():
	def ConvLayer(num_filters, filter_size, input):
		conv = SeparableConv2D(num_filters, filter_size, data_format = "channels_last", padding = "same")(input)
		conv = BatchNormalization()(conv)
		conv = Activation("relu")(conv)
		return conv

	def deConvLayer(num_filters, filter_size, input):
		conv = Conv2DTranspose(num_filters, filter_size, data_format = "channels_last", padding = "same")(input)
		conv = BatchNormalization()(conv)
		conv = Activation("relu")(conv)
		return conv

	visible = Input(shape_of_input)
	conv1 = ConvLayer(16, (7,7), visible)
	pool1 = MaxPooling2D((2,2))(conv1)
	conv2 = ConvLayer(32, (5,5), pool1)
	deconv3 = deConvLayer(8, (5,5), conv2)
	upscale3 = UpSampling2D((2,2))(deconv3)
	deconv4 = deConvLayer(1, (7,7), upscale3)
	model = Model(inputs = visible, outputs = deconv4)
	return model

stereos, depths, orig_stereos, orig_depths = import_data(True)

stereo_model = createModel(True)
print(stereo_model.summary())
stereo_model.compile(optimizer = Adam(0.02, 0.005), loss = "binary_crossentropy")

stereo_model.fit(x = stereos, y = depths, batch_size = 32, epochs = 15, validation_split = 0.1)

for i in range(2):
	predict(100 + i, True)
