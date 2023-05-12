import numpy as np
from keras.preprocessing import image
import tensorflow as tf
tf.reset_default_graph()
sess = tf.Session()
def check(input_img):
    print(" your image is : " + input_img)
    print(input_img)

    tf.reset_default_graph()
    sess = tf.Session()

    saved_model = tf.keras.models.load_model("model/VGG_model.h5")

    img = image.load_img("images/" + input_img, target_size=(224, 224))
    img = np.asarray(img)
    print(img)

    img = np.expand_dims(img, axis=0)

    print(img)
    output = saved_model.predict(img)

    print(output)
    if output[0][0] == 1:
        status = True
    else:
        status = False

    print(status)
    sess.close()
    return status

