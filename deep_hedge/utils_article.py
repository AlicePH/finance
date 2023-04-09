"""
Taken from the article "No-Transaction Band Network" and slightly modified.
"""
import torch
import torch.nn.functional as fn
from torch.distributions.normal import Normal
from torch.nn import Linear, ReLU

def european_option_delta(log_moneyness, time_expiry, volatility):
    s, t, v = map(torch.as_tensor, (log_moneyness, time_expiry, volatility))
    normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
    return normal.cdf((s + (v ** 2 / 2) * t) / (v * torch.sqrt(t)))


def clamp(x, lower, upper):
    x = torch.min(torch.max(x, lower), upper)
    x = torch.where(lower < upper, x, (lower + upper) / 2)
    return x

class MultiLayerPerceptron(torch.nn.ModuleList):
    def __init__(self, in_features, out_features, n_layers=4, n_units=32):
        super().__init__()
        for n in range(n_layers):
            i = in_features if n == 0 else n_units
            self.append(Linear(i, n_units))
            self.append(ReLU())
        self.append(Linear(n_units, out_features))

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

class NoTransactionBandNet(torch.nn.Module):

    def __init__(self, in_features=3):
        super().__init__()
        self.mlp = MultiLayerPerceptron(in_features, 2)

    def forward(self, x, prev):
        no_cost_delta = european_option_delta(x[:, 0], x[:, 1], x[:, 2])
        x = x.to(torch.float32)
        band_width = self.mlp(x)
        lower = no_cost_delta - fn.leaky_relu(band_width[:, 0])
        upper = no_cost_delta + fn.leaky_relu(band_width[:, 1])
        
        hedge = clamp(prev, lower, upper)
        
        return hedge

class FeedForwardNet(torch.nn.Module):

    def __init__(self, in_features=3):
        super().__init__()

        self.mlp = MultiLayerPerceptron(in_features + 1, 1)

    def forward(self, x, prev):
        no_cost_delta = european_option_delta(x[:, 0], x[:, 1], x[:, 2])
        x = torch.cat((x, prev.reshape(-1, 1)), 1)
        x = x.to(torch.float32)
        x = self.mlp(x).reshape(-1)
        x = torch.tanh(x)
        hedge = no_cost_delta + x 

        return hedge

def generate_geometric_brownian_motion_article(
    n_paths, maturity=30 / 365, dt=1 / 365, volatility=0.2, device=None
) -> torch.Tensor:
 
    randn = torch.randn((int(maturity / dt), n_paths), device=device)
    randn[0, :] = 0.0
    bm = volatility * (dt ** 0.5) * randn.cumsum(0)
    t = torch.linspace(0, maturity, int(maturity / dt))[:, None].to(bm)
    return torch.exp(bm - (volatility ** 2) * t / 2)



