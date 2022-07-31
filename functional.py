from torch.nn.functional import softmax
from torch import bmm, cat, transpose, reshape, matmul
import random

__all__ = ['multi_head_attention']


#################################################
# Multi Head Attention 
#################################################

def multi_head_attention(q, k, v):
  """A differentiable multi head attention function.

  Args:
    q (torch.Tensor): The query embedding.
      Has shape `(batch_size, num_heads, sequence_size, head_emb_dim)`.
    k (torch.Tensor): The key embedding.
      Has shape `(batch_size, num_heads, sequence_size, head_emb_dim)`.
    v (torch.Tensor): The value embedding.
      Has shape `(batch_size, num_heads, sequence_size, head_emb_dim)`.

  Returns:
    y (torch.Tensor): The multi head attention output.
      Has shape `(batch_size, sequence_size, num_heads * head_emb_dim)`.
  """
 
  batch_size, num_heads, sequence_size, d = q.shape

  # transpose to shape (num_heads, batch_size, sequence_size, head_emb_dim)
  q_ = q.transpose(0, 1)
  k_ = k.transpose(0, 1)
  v_ = v.transpose(0, 1)
  y_list = []

  for (q_h, k_h, v_h) in zip(q_, k_, v_):
    # randomlist = random.sample(range(0,q_h.size(2)),1)
    # randomlist.sort()
    # q_h = q_h[:,:,randomlist]
    # k_h = k_h[:,:,randomlist]
    Att = softmax(bmm(q_h, k_h.transpose(-2, -1)) * (d ** -0.5), dim=-1)
    y_list.append(bmm(Att, v_h))

  y = cat(y_list, dim=-1)
  
  return y
