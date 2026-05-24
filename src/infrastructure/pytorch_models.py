import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
from src.domain.interfaces import IAnomalyModel

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, emb: int = 64):
        super().__init__()
        self.encoder = nn.LSTM(n_features, emb, batch_first=True)
        self.decoder = nn.LSTM(emb, 128, batch_first=True)
        self.output = nn.Linear(128, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        out, _ = self.decoder(h.transpose(0, 1).repeat(1, x.shape[1], 1))
        return self.output(out)

class GRUAutoencoder(nn.Module):
    def __init__(self, n_features: int, emb: int = 64):
        super().__init__()
        self.encoder = nn.GRU(n_features, emb, batch_first=True)
        self.decoder = nn.GRU(emb, 128, batch_first=True)
        self.output = nn.Linear(128, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.encoder(x)
        out, _ = self.decoder(h.transpose(0, 1).repeat(1, x.shape[1], 1))
        return self.output(out)

class Conv1DAutoencoder(nn.Module):
    def __init__(self, n_features: int, emb: int = 32):
        super().__init__()
        self.emb = emb
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, emb, 3, padding=1), nn.AdaptiveAvgPool1d(1)
        )
        self.decoder = nn.Sequential(nn.Linear(emb, emb * 10), nn.ReLU())
        self.output = nn.Linear(emb, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoder(x.permute(0, 2, 1)).squeeze(-1)
        d = self.decoder(e).view(-1, 10, self.emb)
        return self.output(d)

class VAEAutoencoder(nn.Module):
    def __init__(self, n_features: int, latent: int = 32):
        super().__init__()
        self.encoder_lstm = nn.LSTM(n_features, 64, batch_first=True)
        self.fc_mu = nn.Linear(64, latent)
        self.fc_logvar = nn.Linear(64, latent)
        self.decoder_fc = nn.Linear(latent, 64)
        self.decoder_lstm = nn.LSTM(64, 128, batch_first=True)
        self.output = nn.Linear(128, n_features)
        self.is_vae = True

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, (h, _) = self.encoder_lstm(x)
        mu, logvar = self.fc_mu(h[-1]), self.fc_logvar(h[-1])
        z = self.reparam(mu, logvar)
        dec_in = self.decoder_fc(z).unsqueeze(1).repeat(1, x.shape[1], 1)
        out, _ = self.decoder_lstm(dec_in)
        return self.output(out), mu, logvar

class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.decoder = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(self.input_proj(x)))

class TCNAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        layers = []
        for i in range(3):
            dil = 2**i
            in_ch = n_features if i == 0 else hidden
            layers.append(nn.Conv1d(in_ch, hidden, 3, padding=dil, dilation=dil))
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden, 32, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, n_features, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x.permute(0, 2, 1))
        return self.decoder(enc).permute(0, 2, 1)

class NeuralModelAdapter(IAnomalyModel):
    def __init__(self, module: nn.Module, device: torch.device):
        self._module = module.to(device)
        self._module.eval()
        self._device = device

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tensor_x = torch.tensor(data, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            out = self._module(tensor_x)
            if isinstance(out, tuple):
                out = out[0]
            diff = torch.abs(out.cpu() - torch.tensor(data, dtype=torch.float32))
            f_errors = torch.mean(diff, dim=1).numpy()
            errors = torch.mean(diff, dim=(1, 2)).numpy()
        return f_errors, errors