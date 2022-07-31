from nn import *
from torch.nn import Module, Linear, LayerNorm, Sequential, ModuleList
from torch import cat, stack
from vit_helpers import init_vit_weights

__all__ = ['ViT']


#################################################
# Vision Transformer
#################################################
class ViT(Module):
    def __init__(self, img_dim, patch_dim, in_chans, num_classes, dim, depth, num_heads, init_std):
        """Creates a Vision Transformer model

        Args:
          img_dim (int): Image dim, we use only squared images so the total
            image size is (in_chans, img_dim, img_dim).
          patch_dim (int): Patch dim, we use only squared patches so the total
            patch size is (patch_dim, patch_dim).
          in_chans (int): Number of channels in the input image.
          num_classes (int): Number of classes.
          dim (int): The PatchEmbedding output dimension.
          depth (int): The number of transformer blocks in the model.
          num_heads (int): Number of attention heads.
          init_std (float): the standard deviation of the truncated normal distribution used for initialization.
        """
        super().__init__()

        self.patch_dim = patch_dim
        self.patch_embed = PatchEmbedding(patch_dim, in_chans, dim)
        sequence_size = self.patch_embed.get_sequence_size(img_dim, patch_dim)
        self.pos_embed = PositionalEmbedding(sequence_size, dim, init_std)
        self.cls_token = CLSToken(dim, init_std)
        self.blocks = ModuleList([TransformerBlock(dim=dim, num_heads=num_heads) for i in range(depth)])
        self.norm = LayerNorm(dim)

        # Initialize weights
        self.apply(init_vit_weights)

    def forward(self, x):
        """Computes the forward function of a vision transformer model

        Args:
          x (torch.Tensor): The input images.
            Has shape `(batch_size, in_chans, img_dim, img_dim)`, we use only squared images.

        Returns:
          y (torch.Tensor): The output classification tensor.
            Has shape `(batch_size, num_classes)`.
        """

        batch_size, c, h, w = x.shape
        # patch linear embedding
        x = self.patch_embed(x)
        # add the [CLS] token to the embed patch tokens
        cls_token_list = []
        for i in range(batch_size):
            cls_token_list.append(self.cls_token())
        cls_tokens = cat(cls_token_list, dim=0)
        x = cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        y = x[:, 0]
        return y

# from nn import *
# from torch.nn import Module, Linear, LayerNorm, Sequential, ModuleList
# from torch import cat, stack
# from vit_helpers import init_vit_weights
# '''Q5: Why we are increasing the dimension (from 49 to 64) of each patch after patch embedding? Can't we reduce?'''
#
# __all__ = ['ViT']
#
# #################################################
# # Vision Transformer
# #################################################
# class ViT(Module):
#   def __init__(self, img_dim, patch_dim, in_chans, num_classes, dim, depth, num_heads, init_std):
#     """Creates a Vision Transformer model
#
#     Args:
#       img_dim (int): Image dim, we use only squared images so the total
#         image size is (in_chans, img_dim, img_dim).
#       num_levels (int): Denotes the number of levels in wavelet transform
#       patch_dim (int): Patch dim, we use only squared patches so the total
#         patch size is (patch_dim, patch_dim).
#       in_chans (int): Number of channels in the input image.
#       num_classes (int): Number of classes.
#       dim (int): The PatchEmbedding output dimension.
#       depth (int): The number of transformer blocks in the model.
#       num_heads (int): Number of attention heads.
#       init_std (float): the standard deviation of the truncated normal distribution used for initialization.
#     """
#     super().__init__()
#
#     self.patch_dim = patch_dim
#     self.patch_embed = PatchEmbedding(patch_dim, in_chans, dim)
#     sequence_size =  self.patch_embed.get_sequence_size(img_dim, patch_dim) #3*num_levels+1
#     self.pos_embed = PositionalEmbedding(sequence_size, dim, init_std)
#     self.cls_token = CLSToken(dim, init_std)
#     self.blocks = ModuleList([TransformerBlock(dim=dim, num_heads=num_heads) for i in range(depth)])
#     self.norm = LayerNorm(dim)
#
#     '''Q7: How exactly this apply is working?'''
#     # Initialize weights
#     self.apply(init_vit_weights)
#
#   def forward(self, x):
#     """Computes the forward function of a vision transformer model
#
#     Args:
#       x (torch.Tensor): The input images.
#         Has shape `(batch_size, in_chans, img_dim, img_dim)`, we use only squared images.
#
#     Returns:
#       y (torch.Tensor): The output classification tensor.
#         Has shape `(batch_size, num_classes)`.
#     """
#
#     batch_size, c, h, w = x.shape
#     # patch linear embedding
#     x = self.patch_embed(x)
#     # add the [CLS] token to the embed patch tokens
#     cls_token_list = []
#     for i in range (batch_size):
#       cls_token_list.append(self.cls_token())
#     cls_tokens = cat(cls_token_list, dim=0)
#     x = cat((cls_tokens, x), dim=1)
#
#     for blk in self.blocks:
#         x = blk(x)
#     x = self.norm(x)
#     y = x[:, 0]
#     return y
#
#
#
#
#
#
#
