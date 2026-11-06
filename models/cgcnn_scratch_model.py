import torch
from torch import nn

class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        n, m = nbr_idx.shape
        nbr_atom = atom_fea[nbr_idx, :]
        combined = torch.cat([atom_fea.unsqueeze(1).expand(n, m, -1), nbr_atom, nbr_fea], dim=2)
        z = self.fc_full(combined)
        z = self.bn1(z.reshape(-1, 2 * self.atom_fea_len)).reshape(n, m, -1)
        gate, core = z.chunk(2, dim=2)
        nbr_sum = (self.sigmoid(gate) * self.softplus1(core)).sum(dim=1)
        return self.softplus2(atom_fea + self.bn2(nbr_sum))

class CGCNNBackbone(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1):
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

    def forward(self, atom, nbr, nbr_idx, crystal_idx):
        atom = self.embedding(atom)
        for conv in self.convs:
            atom = conv(atom, nbr, nbr_idx)
        pooled = torch.cat([atom[idx].mean(0, keepdim=True) for idx in crystal_idx], dim=0)
        out = self.conv_to_fc(self.conv_to_fc_softplus(pooled))
        out = self.conv_to_fc_softplus(out)
        if hasattr(self, "fcs"):
            for fc, act in zip(self.fcs, self.softpluses):
                out = act(fc(out))
        return out

class CGCNNMultiGas(nn.Module):
    def __init__(self, graph_config, head_hidden=64, dropout=0.1):
        super().__init__()
        self.backbone = CGCNNBackbone(**graph_config)
        self.output_heads = nn.ModuleDict({
            gas: nn.Sequential(nn.Linear(graph_config["h_fea_len"] + 1, head_hidden),
                               nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1))
            for gas in ["CO2", "N2"]
        })

    def forward(self, atom, nbr, nbr_idx, crystal_idx, pressure, gas_index, sample_to_crystal=None):
        structure = self.backbone(atom, nbr, nbr_idx, crystal_idx)
        if sample_to_crystal is not None:
            structure = structure[sample_to_crystal]
        joint = torch.cat([structure, pressure], dim=1)
        co2 = self.output_heads["CO2"](joint).view(-1)
        n2 = self.output_heads["N2"](joint).view(-1)
        return torch.where(gas_index == 0, co2, n2)

def load_cgcnn(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CGCNNMultiGas(checkpoint["graph_config"],
                          checkpoint["head_hidden"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    if checkpoint.get("params", {}).get("finetune_mode") == "head":
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    return model.to(device).eval(), checkpoint["preprocessor_state"], checkpoint["atom_init"]
