import numpy as np
from matplotlib import pyplot as plt



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_data(data_sample):
    from qr_torch.qrdata import Label
    #plt.imshow(np.swapaxes(data_sample[0].numpy(), 0, 2))
    plt.imshow(np.transpose(data_sample[0].numpy(), (1, 2, 0)))

    plt.title('y = ' + Label(data_sample[1]).name)