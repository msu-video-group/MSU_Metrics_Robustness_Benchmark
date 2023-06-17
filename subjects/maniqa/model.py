from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
from maniqa_models.maniqa import MANIQA
from config import Config
sys.path.remove(str(Path(__file__).parent))


class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        
        config = Config({
            # model
            "patch_size": 8,
            "img_size": 224,
            "embed_dim": 768,
            "dim_mlp": 768,
            "num_heads": [4, 4],
            "window_size": 4,
            "depths": [2, 2],
            "num_outputs": 1,
            "num_tab": 2,
            "scale": 0.8,

        })
        
        model = MANIQA(
            embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
            patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
            depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale
        )
        
        model.load_state_dict(torch.load(model_path), strict=False)
        model.eval().to(device)
        
        self.model = model
    
    def forward(self, image, inference=False):
        out = self.model(
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.Resize([224, 224])(image))
        )
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out
        
        