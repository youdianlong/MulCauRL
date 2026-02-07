import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms


def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    try:
        from conch.open_clip_custom import create_model_from_pretrained  # noqa: F401
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH


def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    try:
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH



class VisionMambaPatchEncoder(nn.Module):

    def __init__(self, vmamba_dir: str, img_size: int = 224, pool: str = "cls"):
        super().__init__()
        self.base = AutoModel.from_pretrained(vmamba_dir, trust_remote_code=True)
        self.img_size = img_size
        assert pool in ("cls", "mean")
        self.pool = pool

    @torch.no_grad()
    def _resize(self, x):
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        return x

    def forward(self, x):  # x: [B,3,H,W]
        x = self._resize(x)
        out = self.base(x)

        if isinstance(out, tuple):
            hs = out[0]
        elif hasattr(out, "last_hidden_state"):
            hs = out.last_hidden_state
        elif isinstance(out, torch.Tensor):
            hs = out
        else:
            raise RuntimeError(f"Unexpected Vision Mamba output type: {type(out)}")

        if hs.ndim == 3:
            if self.pool == "cls":
                z = hs[:, 0]                    # [B, D]
            else:
                z = hs.mean(dim=1)              # [B, D]
        elif hs.ndim == 2:
            # [B, D] 
            z = hs
        elif hs.ndim == 1:
            z = hs.unsqueeze(0)
        else:
            z = hs.view(hs.size(0), -1)

        if z.ndim != 2:
            z = z.view(z.size(0), -1)

        return z  


class VMambaPerPatch(nn.Module):

    def __init__(self, vmamba_dir: str, img_size: int = 224, pool: str = "cls"):
        super().__init__()
        self.encoder = VisionMambaPatchEncoder(vmamba_dir=vmamba_dir, img_size=img_size, pool=pool)

    def forward(self, images: torch.Tensor):
        return self.encoder(images)   # [B, D]


        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'conch_v1_5':
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use CONCH v1.5")
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, _ = titan.return_conch()
        assert target_img_size == 448, 'TITAN is used with 448x448 CONCH v1.5 features'
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms