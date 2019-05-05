import os
import tensorflow as tf

label_bytes = 1
image_bytes = 32 * 32 * 3


def preprocess(sample):
    sample_bytes = tf.io.decode_raw(sample, tf.uint8)
    label = tf.cast(tf.strided_slice(sample_bytes, [0], [label_bytes]), tf.int32)
    image = tf.strided_slice(sample_bytes, [label_bytes], [label_bytes + image_bytes])
    image = tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0])
    image = tf.cast(image, tf.float32) / 255
    return image, label


def get_train_dataset(dir_input, batch_size):
    filenames = tf.io.gfile.glob(os.path.join(dir_input, "data_batch_*.bin"))
    train_dataset = tf.data.FixedLengthRecordDataset(filenames, label_bytes + image_bytes)
    train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def augmentation(image, label):
        image = tf.image.random_flip_left_right(image)
        return image, label

    train_dataset = train_dataset.map(augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=32).repeat().batch(batch_size)

    return train_dataset


def get_test_dataset(dir_input, batch_size):
    filenames = tf.io.gfile.glob(os.path.join(dir_input, "test_batch.bin"))
    test_dataset = tf.data.FixedLengthRecordDataset(filenames, label_bytes + image_bytes)
    test_dataset = test_dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)
    return test_dataset
