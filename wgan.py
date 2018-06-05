import click
import tensorflow as tf
from keras import backend as K

import dataset
from wgan import model as Model
from wgan import logging


def echo(*args):
    click.secho(' '.join(str(arg) for arg in args), fg='green', err=True)


@click.group()
def main():
    pass


@main.command()
@click.option('--name', help='model name', required=True)
@click.option('--critics', type=int, default=2)
@click.option('--clip', type=float, default=0.1)
@click.option('--batch-size', type=int, default=256)
@click.option('--epochs', type=int, default=500)
@click.option('--out', type=str, default='out/')
@click.option('--verbose', type=int, default=1)
def train(name, critics, clip, batch_size, epochs, out, verbose):

    # paths
    log_path = "logs/{}.json".format(name)
    out_path = "snapshots/" + name + ".{epoch:06d}.h5"
    echo('log path', log_path)
    echo('out path', out_path)

    # init
    echo('train', locals())
    logging.info(log_path, {'train': locals()})
    session = tf.Session('')
    K.set_session(session)
    K.set_learning_phase(1)

    # dataset
    echo('dataset loading...')
    data = dataset.batch_generator(batch_size)

    # model building
    echo('model building...')
    models = Model.build()
    echo('- Generator:')
    models[0].summary()
    echo('- Critic:')
    models[1].summary()

    # training
    echo('start learning...')
    Model.train(models,
                data,
                epochs,
                batch_size,
                critics,
                clip,
                out=out,
                log=lambda data: logging.info(log_path, data))


if __name__ == '__main__':
    main()
