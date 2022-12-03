import torch
import sys

model = torch.load(sys.argv[1], map_location=torch.device('cpu'))
model.eval()

torchscript_model = torch.jit.script(model)
torch.jit.save(torchscript_model, "./model.pt")