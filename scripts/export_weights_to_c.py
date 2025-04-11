import torch
import numpy as np
import os
import sys

# Add project root to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.basic_denoiser import BasicDenoiser

def float_to_fixed(val, scale=256):
    return int(round(val * scale))

def tensor_to_fixed_array(tensor, scale=256):
    arr = tensor.cpu().numpy().flatten()
    return [float_to_fixed(v, scale) for v in arr]

# Load model
model = BasicDenoiser()
model.load_state_dict(torch.load("models/denoiser.pth"))
model.eval()

# Output file
os.makedirs("export", exist_ok=True)
header_path = "export/model_weights.h"

with open(header_path, "w") as f:
    f.write("// Auto-generated fixed-point weights\n\n")
    f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
    f.write("#include <stdint.h>\n\n")

    for name, param in model.named_parameters():
        fixed_vals = tensor_to_fixed_array(param.data)
        shape = param.shape
        var_name = name.replace(".", "_")

        f.write(f"// {name} shape: {list(shape)}\n")
        f.write(f"int16_t {var_name}[{len(fixed_vals)}] = {{\n")
        for i, val in enumerate(fixed_vals):
            f.write(f"{val}, ")
            if (i + 1) % 8 == 0:
                f.write("\n")
        f.write("};\n\n")

    f.write("#endif // MODEL_WEIGHTS_H\n")

print(f"✅ Fixed-point weights exported to {header_path}")