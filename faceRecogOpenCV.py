import face_recognition
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import PIL 

def main():
    faceRecognition()

def faceRecognition():
    face_locations = []
    FILE_PATH="sample.jpg"
    face_image = face_recognition.load_image_file(FILE_PATH)

    face_locations = face_recognition.face_locations(face_image,model="hog")
    for face_location in face_locations:
        top, right, bottom, left = face_location
        img = cv2.imread(FILE_PATH)
        
        img_trim = img[top:bottom,left:right]
        img_edge = cv2.Canny(img_trim,30,150)
        dst = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        dst = cv2.resize(dst,dsize=(256,256))
        
        print(f"{img_trim.shape}->{dst.shape}")

        cv2.imwrite("result2.jpg",dst)    

        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.imwrite("result.jpg",img)
        generator = Generator()
        discriminator = Discriminator()

        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)
        checkpoint.restore(tf.train.latest_checkpoint("training_checkpoints"))

        #np_image_data = load("result2.jpg")

        #test_dataset = tf.data.Dataset.list_files('facades/test/*.jpg')
        test_dataset = tf.data.Dataset.list_files('result2.jpg')
        test_dataset = test_dataset.map(load_image_test)
        test_dataset = test_dataset.batch(1)

        model=generator
        for inp in test_dataset.take(1):
            #print(f"{inp.shape}")
            prediction = model(inp, training=True)
            image = tensor_to_image(prediction[0])
            image.show()
            image.save("result6.jpg")
           # cv2.imwrite("result6.jpg",image)
            plt.imshow(prediction[0])
            plt.axis('off')
            plt.show()
            #plt.savefig("result6.jpg", bbox_inches='tight')
        
        org_img = cv2.imread(FILE_PATH)
        new_img = cv2.imread("result6.jpg")
        dx=left
        dy=top
        h,w = img_trim.shape[:2]
        new_img = cv2.resize(new_img,dsize=(h,w))
        org_img[dy:dy+h,dx:dx+w] = new_img
        cv2.imwrite("result7.jpg",org_img)


def tensor_to_image(tensor):
    #tensor = tensor*255
    tensor = (tensor + 1) * 127.5
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def resize(input_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  #real_image = tf.image.resize(real_image, [height, width],
  #                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image

def load_image_test(image_file):
  input_image = load(image_file)
  #input_image= resize(input_image, 256, 256)
  input_image = normalize(input_image)

  return input_image


def normalize(input_image):
  input_image = (input_image / 127.5) - 1
  #real_image = (real_image / 127.5) - 1

  return input_image

def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)
  # Convert both images to float32 tensors
  input_image = tf.cast(image, tf.float32)
  #real_image = tf.cast(real_image, tf.float32)

  return input_image


def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])
  
  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  OUTPUT_CHANNELS = 3
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.ReLU())

  return result




if __name__ == "__main__":
    main()
