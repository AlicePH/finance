"""
Original source of the code: https://github.com/mmssss/noa/blob/feature/quant/docs/quant/heston_sim.ipynb
"""
import torch
import math

def noncentral_chisquare(df: torch.Tensor, nonc: torch.Tensor) -> torch.Tensor:
    PSI_CRIT = 1.5
    m = df + nonc
    s2 = 2*df + 4*nonc
    psi = s2 / m.pow(2)
    psi_inv = 1 / psi
    b2 = 2*psi_inv - 1 + (2*psi_inv).sqrt() * (2*psi_inv - 1).sqrt()
    a = m / (1 + b2)
    sample_quad = a * (b2.sqrt() + torch.randn_like(a)).pow(2)
    p = (psi - 1) / (psi + 1)
    beta = (1 - p) / m
    rand = torch.rand_like(p)
    sample_exp = torch.where((p < rand) & (rand <= 1),
                             beta.pow(-1)*torch.log((1-p)/(1-rand)),
                             torch.zeros_like(rand))
    return torch.where(psi <= PSI_CRIT, sample_quad, sample_exp)


def generate_cir(n_paths: int, n_steps: int, dt: float, init_state: torch.Tensor,
                 kappa: float, theta: float, eps: float) -> torch.Tensor:
    if init_state.shape != torch.Size((n_paths,)):
        raise ValueError('Shape of `init_state` must be (n_paths,)')

    paths = torch.empty((n_paths, n_steps + 1), dtype=init_state.dtype)
    paths[:, 0] = init_state

    delta = 4 * kappa * theta / (eps * eps) * torch.ones_like(init_state)
    exp = math.exp(-kappa*dt)
    c_bar = 1 / (4*kappa) * eps * eps * (1 - exp)
    for i in range (0, n_steps):
        v_cur = paths[:, i]
        kappa_bar = v_cur * 4*kappa*exp / (eps * eps * (1 - exp))
        v_next = c_bar * noncentral_chisquare(delta, kappa_bar)
        paths[:, i+1] = v_next
    return paths


def generate_heston(n_paths: int, n_steps: int, dt: float,
                    init_state_price: torch.Tensor,
                    init_state_var: torch.Tensor,
                    kappa: float, theta: float, eps: float, rho: float, drift: float):
    if init_state_price.shape != torch.Size((n_paths,)):
        raise ValueError('Shape of `init_state_price` must be (n_paths,)')
    if init_state_var.shape != torch.Size((n_paths,)):
        raise ValueError('Shape of `init_state_var` must be (n_paths,)')

    gamma2 = 0.5
    if rho > 0:  
        L = rho*dt*(kappa/eps - 0.5*rho)
        R = 2*kappa/(eps*eps*(1 - math.exp(-kappa*dt))) - rho/eps
        if R<=0 or L==0 or (L<0 and R>=0):
            pass
        elif L > 0:
            gamma2 = min(0.5, R / L * 0.9)  
    gamma1 = 1.0 - gamma2

    k0 = -rho * kappa * theta * dt / eps
    k1 = gamma1 * dt * (kappa * rho / eps - 0.5) - rho / eps
    k2 = gamma2 * dt * (kappa * rho / eps - 0.5) + rho / eps
    k3 = gamma1 * dt * (1 - rho * rho)
    k4 = gamma2 * dt * (1 - rho * rho)

    var = generate_cir(n_paths, n_steps, dt, init_state_var, kappa, theta, eps)
    log_paths = torch.empty((n_paths, n_steps + 1), dtype=init_state_price.dtype)
    log_paths[:, 0] = init_state_price.log()

    for i in range(0, n_steps):
        v_i = var[:, i]
        v_next = var[:, i+1]
        next_vals = drift*dt + log_paths[:, i] + k0 + k1*v_i + k2*v_next + \
            torch.sqrt(k3*v_i + k4*v_next) * torch.randn_like(v_i)
        log_paths[:, i+1] = next_vals
    return log_paths.exp(), var
