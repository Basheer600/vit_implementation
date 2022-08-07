import torch  # The torch package includes data structure for multi-dimensional tensors and mathematical operation over these are defined.
import torch.nn.functional as F  # This package has functional classes which are similar to torch.nn.
import torchvision  # The torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision.
from torchvision import transforms  # transforms: Transforms are common image transformations available in the "torchvision.transforms" module.
from models import ViT
import time  #As the name suggests Python time module allows to work with time in Python. It allows functionality like getting the current time, pausing the Program from executing, etc.
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.fftpack import fft, dct
import pywt

'''The torch.nn namespace provides all the building blocks you need, to build and train your own neural network.
https://pytorch.org/docs/stable/nn.html#containers

torch.optim: This package is used to implement various optimization algorithm.'''

def train_epoch(model, optimizer, data_loader, loss_history, device):
  # torch.autograd.set_detect_anomaly(True)
  total_samples = len(data_loader.dataset)

  '''model.train() tells your model that you are training the model. So effectively layers like dropout, batchnorm etc.
   which behave different on the train and test procedures know what is going on and hence can behave accordingly.'''
  model.train()

  for i, (data, target) in enumerate(data_loader):
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor s to zero.
    output = F.log_softmax(model(data), dim=1) # Applies a softmax followed by a logarithm
    loss = F.nll_loss(output, target) # The negative log likelihood loss.
    loss.backward() # The backward() method is used to compute the gradient during the backward pass in a neural network.
    optimizer.step() # Performs a single optimization step (parameter update).

    if i % 100 == 0:
      print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
            ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
            '{:6.4f}'.format(loss.item()))
      loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history, device):
  model.eval()

  total_samples = len(data_loader.dataset)
  correct_samples = 0
  total_loss = 0

  with torch.no_grad():
    for data, target in data_loader:
      data = data.to(device)
      target = target.to(device)
      output = F.log_softmax(model(data), dim=1)
      loss = F.nll_loss(output, target, reduction='sum')
      _, pred = torch.max(output, dim=1)

      total_loss += loss.item()
      correct_samples += pred.eq(target).sum()

  avg_loss = total_loss / total_samples
  loss_history.append(avg_loss)
  print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
        '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
        '{:5}'.format(total_samples) + ' (' +
        '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

def dct2(a):
  return scipy.fftpack.dct(scipy.fftpack.dct(a.numpy(), axis=0, norm='ortho'), axis=1, norm='ortho')

def train_vit():
  # A torch.device is an object representing the device on which a torch.Tensor is or will be allocated.
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # device = 'cpu'
  start_time = time.time()

  BATCH_SIZE_TRAIN = 100
  BATCH_SIZE_TEST = 1000
  N_EPOCHS = 6

  DOWNLOAD_PATH = './data/CIFAR100'

  # Load data
  ''' transforms: Transforms are common image transformations available in the "torchvision.transforms" module. They can
   be chained together using Compose.

   transforms.Normalize: Normalize a tensor image with mean and standard deviation. This transform does not support PIL
   Image. Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels, this transform will normalize
   each channel of the input torch.*Tensor i.e., output[channel] = (input[channel] - mean[channel]) / std[channel]

   transforms.ToTensor: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W)
   in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or
   if the numpy.ndarray has dtype = np.uint8. In the other cases, tensors are returned without scaling.'''
  transform_CIFAR100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

  '''Q1:Why the normalization is for only a particular channel? How the mean and variance values are considered?
     Q2: Where can I find the downloaded MNIST dataset?
     Q3: Why the data values are in the range 0 to 255 even after doing the transform in the below command?
     Q4: What exactly "num_workers" means?
  '''


  '''torchvision.datasets.MNIST(root: str, train: bool = True, transform: Optional[Callable] = None,
  target_transform: Optional[Callable] = None, download: bool = False)

  root (string) – Root directory of dataset where MNIST/raw/train-images-idx3-ubyte and MNIST/raw/t10k-images-idx3-ubyte exist.
  train (bool, optional) – If True, creates dataset from train-images-idx3-ubyte, otherwise from t10k-images-idx3-ubyte.
  download (bool, optional) – If True, downloads the dataset from the internet and puts it in root directory. If dataset
  is already downloaded, it is not downloaded again.
  transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
  target_transform (callable, optional) – A function/transform that takes in the target and transforms it.

  torch.utils.data.DataLoader: torch.utils.data.DataLoader and torch.utils.data.Dataset that allow you to use pre-loaded datasets as well as your own
   data. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
   dataset (Dataset) – dataset from which to load the data.
  batch_size (int, optional) – how many samples per batch to load (default: 1).
  shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
  num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
  '''
  train_set = torchvision.datasets.CIFAR100(DOWNLOAD_PATH, train=True, download=True, transform=transform_CIFAR100)
  # # Computing DCT-2
  # for i in range(train_set.data.size(0)):
  #   train_set.data[i] = torch.tensor(dct2(train_set.data[i]))
  #   plt.imshow(train_set.data[i])

  train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2)

  test_set = torchvision.datasets.CIFAR100(DOWNLOAD_PATH, train=False, download=True, transform=transform_CIFAR100)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=2)

  # Initialize model
  model = ViT(32, 8, 3, 100, 128, 6, 8, .02).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # Train model
  train_loss_history, test_loss_history = [], []
  for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    curr_start_time = time.time()
    train_epoch(model, optimizer, train_loader, train_loss_history, device)
    evaluate(model, test_loader, test_loss_history, device)
    print(f'Epoch {epoch} execution time:', '{:5.2f}'.format((time.time() - curr_start_time) / 60), 'minutes\n')

  print('Execution time:', '{:5.2f}'.format((time.time() - start_time) / 60), 'minutes')


if __name__ == "__main__":
  train_vit()
