import tensorflow as tf

class_names = ['downdog', 'tree', 'warrior1']
img = tf.keras.preprocessing.image.load_img('imgs/Photo on 21-03-24 at 16.07.jpg', target_size=(256, 256))

def load_yoga_model(version='2'):
    model = tf.keras.models.load_model(f'assets/yoga_v{version}.h5')
    return model

def classify_yoga_pose(img,model):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    prediction_class = class_names[predictions.argmax()]
    print('Predictions:', predictions, 'Prediction:', prediction_class)
    return prediction_class, predictions.max()

if __name__ == '__main__':
    img = tf.keras.preprocessing.image.load_img('imgs/Photo on 21-03-24 at 16.07.jpg', target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    model=load_yoga_model(3)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    prediction_class = class_names[predictions.argmax()]
    print(predictions, prediction_class,predictions.max())