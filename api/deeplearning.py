import tensorflow as tf
from keras.models import load_model

graph = tf.get_default_graph()
model = load_model('/home/kiran/Desktop/Plant-Diseases-Recognition/web_app/AlexNetModel.hdf5')
