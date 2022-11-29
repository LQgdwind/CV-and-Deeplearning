from __future__ import division
from collections import namedtuple, deque

class ImageBuffer:
    """Fixed-size buffer to store image tuples."""

    def __init__(self, buffer_size):
        """Initialize a ImageBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.images = namedtuple("Images", field_names=['real_image_X', 'fake_image_X',
                                                        'real_image_Y', 'fake_image_Y'])

    def add(self, real_image_X, fake_image_X, real_image_Y, fake_image_Y):
        """Add a new images to memory."""
        image_pair = self.images(real_image_X, fake_image_X.clone(), real_image_Y, fake_image_Y.clone())
        self.memory.append(image_pair)

    def sample(self):
        """Return a batch of image tuples from memory."""
        return self.memory.popleft()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)