import numpy
import random
import keras.backend as K
from keras import layers, optimizers
from keras.models import Model, Sequential

from wgan.space import Space, Euclidean
from dataset import array2images

X = Space(shape=(32, 32, 3))
Z = Euclidean(shape=(64,))
R = Euclidean(shape=1)


def build():

    def build_critic() -> Sequential:
        """
        X -> R
        """
        model = Sequential()
        model.add(layers.Conv2D(16, kernel_size=2, padding='same', activation='relu',
                  input_shape=X.shape))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(32, kernel_size=3, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(R.shape))
        return model

    def build_generator() -> Sequential:
        """
        Z -> X
        """
        model = Sequential()
        model.add(layers.Dense(128 * 4 * 4, activation='relu', input_shape=Z.shape))
        model.add(layers.Reshape((4, 4, 128)))
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(128, kernel_size=4, padding='same'))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Activation('relu'))
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(64, kernel_size=4, padding='same'))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Activation('relu'))
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(X.shape[2], kernel_size=4, padding='same', activation='tanh'))
        return model

    def wasserstein(y_true, y_pred):
        return K.mean(y_true * y_pred)

    opt = optimizers.RMSprop(lr=0.0005)

    # training critic
    critic = build_critic()
    critic.compile(
            loss=wasserstein,
            optimizer=opt)

    # training generator
    generator = build_generator()
    z = layers.Input(shape=Z.shape)
    x = generator(z)
    valid = critic(x)
    combined = Model(z, valid)

    critic.trainable = False
    combined.compile(
            loss=wasserstein,
            optimizer=opt)

    return generator, critic, combined


def train(models, batch, epochs, batch_size, n_critic, clip, out='out/', log=None):

    generator, critic, combined = models
    if out[-1] != '/':
        out = out + '/'

    y_real = -numpy.ones((batch_size,))
    y_fake = numpy.ones((batch_size,))

    for epoch in range(epochs):

        print(f"Epoch #{epoch+1}")

        for __ in range(100):

            critic.trainable = True
            for _ in range(n_critic):

                idx = random.randrange(len(batch))
                x_real = batch[idx]
                while len(x_real) < batch_size:
                    idx = random.randrange(len(batch))
                    x_real = batch[idx]
                c_real_res = critic.train_on_batch(x_real, y_real)
                del x_real

                z = Z.sampling(batch_size)
                x_fake = generator.predict(z)
                c_fake_res = critic.train_on_batch(x_fake, y_fake)

                for layer in critic.layers:
                    ws = layer.get_weights()
                    layer.set_weights([numpy.clip(w, -clip, clip) for w in ws])

            critic.trainable = False
            g_res = combined.train_on_batch(z, y_real)

        print(f"  Loss: c/real={c_real_res:.8f} c/fake={c_fake_res:.8f} g={g_res:.8f}")
        if log is not None:
            log({
                'epoch': epoch,
                'loss': {
                    'critic': {
                        'real': float(c_real_res),
                        'fake': float(c_fake_res)
                    },
                    'gen': float(g_res)
                }
            })

        imgs = array2images(x_fake)
        for i in range(min(16, len(imgs))):
            path = f"{out}{epoch:06d}.{i:02x}.png"
            imgs[i].save(path)
