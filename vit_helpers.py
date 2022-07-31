from torch.nn import Module, Linear, GELU, GroupNorm, BatchNorm2d, LayerNorm
from torch.nn.init import trunc_normal_, zeros_, constant_, ones_

class Mlp(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        hidden_features = in_features * 4
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def init_vit_weights(module: Module, name: str = ''):
  if isinstance(module, Linear):
    trunc_normal_(module.weight, std=.02)
    zeros_(module.bias)
  elif isinstance(module, (LayerNorm, GroupNorm, BatchNorm2d)):
    zeros_(module.bias)
    ones_(module.weight)




  