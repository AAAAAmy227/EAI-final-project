
import torch
import torch.nn as nn
from tensordict import from_module
from tensordict.nn import CudaGraphModule

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)

device = "cuda"
net = SimpleNet().to(device)
net_inf = SimpleNet().to(device)

# Initialize with different weights
nn.init.constant_(net.fc.weight, 1.0)
nn.init.constant_(net_inf.fc.weight, 0.0)

# Compile net_inf
compiled_inf = torch.compile(net_inf.forward)
graph_inf = CudaGraphModule(compiled_inf)

# Test before sync
x = torch.ones(1, 10, device=device)
out0 = graph_inf(x)
print(f"Output before sync (expected ~0): {out0.item():.4f}")

# Sync using from_module(...).to_module(...)
from_module(net).data.to_module(net_inf)

# Test after sync
out1 = graph_inf(x)
print(f"Output after sync (expected ~10): {out1.item():.4f}")

if abs(out1.item() - 10.0) < 1e-3:
    print("SUCCESS: CudaGraphModule follows weight changes via to_module!")
else:
    print("FAILURE: CudaGraphModule is STUCK with old weights!")
