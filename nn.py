import math
import torch
from torch.nn import init
from functional import multi_head_attention
from torch.nn import Module, Linear, Conv2d, LayerNorm, BatchNorm1d, Parameter
from torch import unbind, reshape, permute, flatten, zeros, empty
from torch.nn.init import trunc_normal_ # Truncated normal distribution
from vit_helpers import Mlp
import pywt


'''__all__: It is a list of strings defining what symbols in a module will be exported when "from <module> import *" is
used on the other module. If the __all__ below is commented out, the default behaviour of import * is to import all
symbols that do not begin with an underscore, from the given namespace. Members that are not mentioned in __all__ are
still accessible from outside the module and can be imported with "from <module> import <member>"'''

__all__ = ['MHSA', 'TransformerBlock', 'PatchEmbedding', 'PositionalEmbedding', 'CLSToken']

#################################################
# Multi Head Self Attention Layer
#################################################

class MHSA(Module):
  def __init__(self, dim, num_heads):
    """Creates a Multi Head Self Attention layer.

    Args:
      dim (int): The input and output dimension (in this implementation Dy=Dq=Dk=Dv=Dx)
      num_heads (int): Number of attention heads.
    """
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.lin_qkv = Linear(dim, dim * 3)
    self.lin = Linear(dim, dim)

  '''Q6: Why the linear transformation is performed for the same dimension?'''

  def forward(self, x):
    """Computes the `MHSA` of the input `x`.

    Args:
      x (torch.Tensor): The input tensor.
        Has shape `(batch_size, sequence_size, dim)`.

    Returns:
      y (torch.Tensor): The output tensor.
        Has shape `(batch_size, sequence_size, dim)`.
    """

    batch_size, sequence_size, C = x.shape

    qkv = self.lin_qkv(x).reshape(batch_size, sequence_size, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    x = multi_head_attention(q, k, v)
    y = self.lin(x)

    return y


#################################################
# Transformer Block
#################################################
class TransformerBlock(Module):
  def __init__(self, dim, num_heads):
    """ Creates a transformer block

    Args:
      dim (int): The input dimension
      num_heads (int): Number of attention heads.

    ***Note*** Do not waste time implementing an MLP. An implementation is
    already provided to you (see vit_helpers->Mlp).
    """
    super().__init__()

    self.norm1 = LayerNorm(dim)
    self.mhsa = MHSA(dim, num_heads)
    self.norm2 = LayerNorm(dim)
    self.mlp = Mlp(in_features=dim, out_features=dim)


  def forward(self, x):
    """Apply a transfomer block on an input `x`.

    Args:
      x (torch.Tensor): The input tensor.
        Has shape `(batch_size, sequence_size, dim)`.

    Returns:
      y (torch.Tensor): The output tensor.
        Has shape `(batch_size, sequence_size, dim)`.
    """
    # x += self.mhsa(self.norm1(x))
    x = x + self.mhsa(self.norm1(x))
    y = self.mlp(self.norm2(x)) + x
    return y




#################################################
# Patch Embedding
#################################################

class PatchEmbedding(Module):
  """ Divide an image into patches and project them to a given dimension.
  """

  def __init__(self, patch_dim, in_chans, dim):
    """Creates a PatchEmbedding layer.

    Args:
      patch_dim (int): Patch dim, we use only squared patches and squared
        images so the total patch size is (patch_dim, patch_dim).
      in_chans (int): Number of channels in the input image
      dim (int): The projection output dimension.
    """
    super().__init__()
    self.patch_dim = patch_dim
    self.in_chans = in_chans
    self.dim = dim
    self.patch_embed = Conv2d(self.in_chans, self.dim, kernel_size=self.patch_dim, stride=self.patch_dim)
    # self.patch_embed1 = Conv2d(self.in_chans, self.dim, kernel_size=self.patch_dim, stride=self.patch_dim)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


  def forward(self, x):
    """Divide an image into patches and project them.

    Args:
      x (torch.Tensor): The input image.
        Has shape `(batch_size, in_chans, img_dim, img_dim)`, we use only squared images.

    Returns:
      y (torch.Tensor): The output tensor.
        Has shape `(batch_size, sequence_size, dim)`.
    """

    batch_size, in_chans, img_dim, img_dim = x.shape
    # num_patches = get_sequence_size(img_dim, self.patch_dim)
    # Computing the wavelet transform coefficients
    dataArray = torch.empty((batch_size, in_chans, img_dim, img_dim))
    for j in range(batch_size):
      for k in range(in_chans):
        wavCoeffs = pywt.wavedec2(x[j][k].cpu(), 'db1', level=2)
        Array, Slice = pywt.coeffs_to_array(wavCoeffs)
        # Coeff1 = pywt.array_to_coeffs(Array, Slice, output_format='wavedecn')
        # Recon = pywt.waverecn(wavCoeffs, 'db1',mode='periodization')
        dataArray[j][k] = torch.tensor(Array)
    dataArray = dataArray.to(self.device)  # The model is on the GPU, but the data is on the CPU. So, you need to send your input tensors to the GPU.
    y = self.patch_embed(dataArray).flatten(2).transpose(1, 2)
    return y

  @staticmethod
  def get_sequence_size(img_dim, patch_dim):
    """Calculate the number of patches

    Args:
      img_dim (int): Image dim, we use only squared images so the total
        image size is (in_chans, img_dim, img_dim).
      patch_dim (int): Patch dim, we use only squared patches so the total
        patch size is (patch_dim, patch_dim).
    """

    num_patches = (img_dim // patch_dim) ** 2
    return num_patches


#################################################
# Positional Embedding
#################################################
class PositionalEmbedding(Module):
  def __init__(self, sequence_size, dim, init_std):
    """Creates a PositionalEmbedding.

    Args:
      sequence_size (int): The sequence size.
      dim (int): The positional embedding dimension.
      init_std (int): The standard deviation of the truncated normal
        distribution used for initialization.

    **Important note:**
    You may not use PyTorch's nn.Embedding layer.
    Instead, create your own tensor to be the learned parameters,
    and don't forget to wrap it with PyTorch's nn.Parameter
    """
    super().__init__()

    self.sequence_size = sequence_size
    self.dim = dim
    self.init_std = init_std
    self.pos_embed = Parameter(zeros(1, self.sequence_size, self.dim))

  def forward(self):
    """Return the positional embedding.

    Returns:
      y (torch.Tensor): The embedding tensor.
        Has shape `(1, sequence_size, dim)`.
    """

    y = trunc_normal_(self.pos_embed, self.init_std)
    return y


#################################################
# CLS Token
#################################################
class CLSToken(Module):
  def __init__(self, dim, init_std):
    """Creates a CLSToken.

    Args:
      dim (int): The token dimension.
      init_std (int): The standard deviation of the truncated normal
        distribution used for initialization.

    **Important note:**
    You may not use PyTorch's nn.Embedding layer.
    Instead, create your own tensor to be the learned parameters,
    and don't forget to wrap it with PyTorch's nn.Parameter
    """
    super().__init__()

    self.dim = dim
    self.init_std = init_std
    self.cls_token = Parameter(zeros(1, 1, dim))

  def forward(self):
    """Returns the Class Token.

    Returns:
      y (torch.Tensor): The token tensor.
        Has shape `(1, 1, dim)`.
    """

    y = trunc_normal_(self.cls_token, self.init_std)
    return y

# import math
# import torch
# from torch.nn import init
# from functional import multi_head_attention
# from torch.nn import Module, Linear, Conv2d, LayerNorm, BatchNorm1d, Parameter
# from torch import unbind, reshape, permute, flatten, zeros, empty
# from torch.nn.init import trunc_normal_ # Truncated normal distribution
# from vit_helpers import Mlp
# import pywt
#
#
# '''__all__: It is a list of strings defining what symbols in a module will be exported when "from <module> import *" is
# used on the other module. If the __all__ below is commented out, the default behaviour of import * is to import all
# symbols that do not begin with an underscore, from the given namespace. Members that are not mentioned in __all__ are
# still accessible from outside the module and can be imported with "from <module> import <member>"'''
#
# __all__ = ['MHSA', 'TransformerBlock', 'PatchEmbedding', 'PositionalEmbedding', 'CLSToken']
#
# #################################################
# # Multi Head Self Attention Layer
# #################################################
#
# class MHSA(Module):
#   def __init__(self, dim, num_heads):
#     """Creates a Multi Head Self Attention layer.
#
#     Args:
#       dim (int): The input and output dimension (in this implementation Dy=Dq=Dk=Dv=Dx)
#       num_heads (int): Number of attention heads.
#     """
#     super().__init__()
#     self.dim = dim
#     self.num_heads = num_heads
#     self.lin_qkv = Linear(dim, dim * 3)
#     self.lin = Linear(dim, dim)
#
#   '''Q6: Why the linear transformation is performed for the same dimension?'''
#
#   def forward(self, x):
#     """Computes the `MHSA` of the input `x`.
#
#     Args:
#       x (torch.Tensor): The input tensor.
#         Has shape `(batch_size, sequence_size, dim)`.
#
#     Returns:
#       y (torch.Tensor): The output tensor.
#         Has shape `(batch_size, sequence_size, dim)`.
#     """
#
#     batch_size, sequence_size, C = x.shape
#
#     qkv = self.lin_qkv(x).reshape(batch_size, sequence_size, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#     q, k, v = qkv[0], qkv[1], qkv[2]
#     x = multi_head_attention(q, k, v)
#     y = self.lin(x)
#
#     return y
#
#
# #################################################
# # Transformer Block
# #################################################
# class TransformerBlock(Module):
#   def __init__(self, dim, num_heads):
#     """ Creates a transformer block
#
#     Args:
#       dim (int): The input dimension
#       num_heads (int): Number of attention heads.
#
#     ***Note*** Do not waste time implementing an MLP. An implementation is
#     already provided to you (see vit_helpers->Mlp).
#     """
#     super().__init__()
#
#     self.norm1 = LayerNorm(dim)
#     self.mhsa = MHSA(dim, num_heads)
#     self.norm2 = LayerNorm(dim)
#     self.mlp = Mlp(in_features=dim, out_features=dim)
#
#
#   def forward(self, x):
#     """Apply a transfomer block on an input `x`.
#
#     Args:
#       x (torch.Tensor): The input tensor.
#         Has shape `(batch_size, sequence_size, dim)`.
#
#     Returns:
#       y (torch.Tensor): The output tensor.
#         Has shape `(batch_size, sequence_size, dim)`.
#     """
#     # x += self.mhsa(self.norm1(x))
#     x = x + self.mhsa(self.norm1(x))
#     y = self.mlp(self.norm2(x)) + x
#     return y
#
#
#
# #################################################
# # Patch Embedding
# #################################################
#
# class PatchEmbedding(Module):
#   """ Divide an image into patches and project them to a given dimension.
#   """
#   def __init__(self, num_levels, in_chans, dim):
#     """Creates a PatchEmbedding layer.
#
#     Args:
#       patch_dim (int): Patch dim, we use only squared patches and squared
#         images so the total patch size is (patch_dim, patch_dim).
#       in_chans (int): Number of channels in the input image
#       dim (int): The projection output dimension.
#     """
#     super().__init__()
#     self.num_levels = num_levels
#     # self.patch_dim = patch_dim
#     # self.in_features = in_chans
#     self.out_features = dim
#     # self.patch_embed = Conv2d(self.in_chans, self.dim, kernel_size=self.patch_dim, stride=self.patch_dim)
#     # self.patch_embed = Linear(self.in_features, self.out_features)
#
#   def forward(self, x):
#     """Divide an image into patches and project them.
#
#     Args:
#       x (torch.Tensor): The input image.
#         Has shape `(batch_size, in_chans, img_dim, img_dim)`, we use only squared images.
#
#     Returns:
#       y (torch.Tensor): The output tensor.
#         Has shape `(batch_size, sequence_size, dim)`.
#     """
#
#     batch_size, in_chans, img_dim, img_dim = x.shape
#     # num_patches = get_sequence_size(img_dim, self.patch_dim)
#     '''torch.transpose(input, dim0, dim1) → Tensor
#     Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
#     '''
#     # y = self.patch_embed(x).flatten(2).transpose(1, 2)
#     dataArray = torch.empty((batch_size, 3*self.num_levels+1,64))
#     for j in range(x.size(0)):
#       Count = 0
#       wavCoeffs = pywt.wavedec2(x[j][0], 'db1', level=2)
#       Array, Slice = pywt.coeffs_to_array(wavCoeffs)
#       for i in range(0,len(wavCoeffs)):
#         Coeff = torch.tensor(wavCoeffs[i])
#         if i>=1:
#           for k in range(0,Coeff.size(0)):
#             self.in_features = Coeff.size(1)*Coeff.size(1)
#             self.patch_embed = Linear(self.in_features, self.out_features)
#             dataArray[j][Count] = self.patch_embed(torch.reshape(Coeff[k], [1, self.in_features]))
#             Count = Count+1
#         else:
#           self.in_features = Coeff.size(1) * Coeff.size(1)
#           self.patch_embed = Linear(self.in_features, self.out_features)
#           dataArray[j][Count] = self.patch_embed(torch.reshape(Coeff,[1,self.in_features]))
#           Count = Count+1
#     return dataArray
#
#
#   @staticmethod
#   def get_sequence_size(img_dim, patch_dim):
#     """Calculate the number of patches
#
#     Args:
#       img_dim (int): Image dim, we use only squared images so the total
#         image size is (in_chans, img_dim, img_dim).
#       patch_dim (int): Patch dim, we use only squared patches so the total
#         patch size is (patch_dim, patch_dim).
#     """
#
#     num_patches = (img_dim // patch_dim) ** 2
#     return num_patches
#
#
#
# #################################################
# # Positional Embedding
# #################################################
# class PositionalEmbedding(Module):
#   def __init__(self, sequence_size, dim, init_std):
#     """Creates a PositionalEmbedding.
#
#     Args:
#       sequence_size (int): The sequence size.
#       dim (int): The positional embedding dimension.
#       init_std (int): The standard deviation of the truncated normal
#         distribution used for initialization.
#
#     **Important note:**
#     You may not use PyTorch's nn.Embedding layer.
#     Instead, create your own tensor to be the learned parameters,
#     and don't forget to wrap it with PyTorch's nn.Parameter
#     """
#     super().__init__()
#
#     self.sequence_size = sequence_size
#     self.dim = dim
#     self.init_std = init_std
#     self.pos_embed = Parameter(zeros(1, self.sequence_size, self.dim))
#
#
#   def forward(self):
#     """Return the positional embedding.
#
#     Returns:
#       y (torch.Tensor): The embedding tensor.
#         Has shape `(1, sequence_size, dim)`.
#     """
#     '''trunc_normal_(tensor: Tensor, mean: float = 0., std: float = 1.) -> Tensor:
#     Fills the input Tensor with values drawn from a truncated normal distribution. The values are effectively drawn from the
#     normal distribution.'''
#     y = trunc_normal_(self.pos_embed, self.init_std)
#     return y
#
#
#
# #################################################
# # CLS Token
# #################################################
# class CLSToken(Module):
#   def __init__(self, dim, init_std):
#     """Creates a CLSToken.
#
#     Args:
#       dim (int): The token dimension.
#       init_std (int): The standard deviation of the truncated normal
#         distribution used for initialization.
#
#     **Important note:**
#     You may not use PyTorch's nn.Embedding layer.
#     Instead, create your own tensor to be the learned parameters,
#     and don't forget to wrap it with PyTorch's nn.Parameter
#     """
#     super().__init__()
#
#     self.dim = dim
#     self.init_std = init_std
#     self.cls_token = Parameter(zeros(1, 1, dim))
#
#
#   def forward(self):
#     """Returns the Class Token.
#
#     Returns:
#       y (torch.Tensor): The token tensor.
#         Has shape `(1, 1, dim)`.
#     """
#
#     y = trunc_normal_(self.cls_token, self.init_std)
#     return y
#
#
#
