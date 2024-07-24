import tensorflow as tf
import datetime

print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
  def __init__(self):
    super().__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

optimizers = [
    {'optimizer': tf.keras.optimizers.Adam(), 'name': 'Adam'},
    {'optimizer': tf.keras.optimizers.RMSprop(), 'name': 'RMSprop'},
    {'optimizer': tf.keras.optimizers.SGD(), 'name': 'SGD'}
]

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(images, labels, model, optimizer):
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels, model):
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for param in optimizers:
  model = MyModel()
  optimizer = param['optimizer']
  log_dir = "logs/fit/" + param['name'] + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  summary_writer = tf.summary.create_file_writer(log_dir)

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

  print("================================")
  print(f"Optimizer: {param['name']}")
  
  with summary_writer.as_default():
    for epoch in range(EPOCHS):
      # Reset the metrics at the start of the next epoch
      train_loss.reset_states()
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()

      for images, labels in train_ds:
        train_step(images, labels, model, optimizer)

      for test_images, test_labels in test_ds:
        test_step(test_images, test_labels, model)

      with tf.summary.record_if(True):
        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
        tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)
        tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
        tf.summary.scalar('test_accuracy', test_accuracy.result(), step=epoch)

      print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result():.2f}, '
        f'Accuracy: {train_accuracy.result() * 100:.2f}, '
        f'Test Loss: {test_loss.result():.2f}, '
        f'Test Accuracy: {test_accuracy.result() * 100:.2f}'
      )
