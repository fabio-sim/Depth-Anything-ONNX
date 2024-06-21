from dataclasses import dataclass
from enum import StrEnum, auto


@dataclass
class Config:
    url: str
    features: int
    out_channels: list[int]


class Encoder(StrEnum):
    vits = auto()
    vitb = auto()
    vitl = auto()
    vitg = auto()

    @property
    def config(self) -> Config:
        return {
            Encoder.vits: Config(
                url="https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
                features=64,
                out_channels=[48, 96, 192, 384],
            ),
            Encoder.vitb: Config(
                url="https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
                features=128,
                out_channels=[96, 192, 384, 768],
            ),
            Encoder.vitl: Config(
                url="https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
                features=256,
                out_channels=[256, 512, 1024, 1024],
            ),
            Encoder.vitg: Config(
                url="Coming Soon",  # TODO
                features=512,
                out_channels=[1536, 1536, 1536, 1536],
            ),
        }[self]
