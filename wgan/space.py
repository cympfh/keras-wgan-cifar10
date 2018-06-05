import numpy


class Space():

    def __init__(self, shape=None):
        self.shape = shape

    def sampling(self):
        raise NotImplemented


class Euclidean(Space):

    def __init__(self, shape):
        super().__init__(shape=shape)

    def sampling(self, batch_size=None):
        """
        >>> R = Euclidean(1)
        >>> R.sampling().shape
        (1,)
        >>> R.sampling(10).shape
        (10, 1)
        """
        if batch_size is None:
            size = self.shape
        elif type(self.shape) == int:
            size = (batch_size, self.shape)
        else:
            size = (batch_size,) + tuple(self.shape)
        return numpy.random.normal(size=size)
