import hashlib
import hydra
import torch
from mineclip import MineCLIP


@hydra.main(config_name="conf", config_path=".", version_base="1.1")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = cfg.pop("ckpt")

    assert (
        hashlib.md5(open(ckpt.path, "rb").read()).hexdigest() == ckpt.checksum
    ), "broken ckpt"

    model = MineCLIP(**cfg).to(device)
    model.load_ckpt(ckpt.path, strict=True)
    print("Successfully loaded ckpt")
