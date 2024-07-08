from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Optional


@dataclass
class Config:
    url: str
    features: int
    out_channels: list[int]


class Metric(StrEnum):
    indoor = auto()
    outdoor = auto()


class Encoder(StrEnum):
    vits = auto()
    vitb = auto()
    vitl = auto()
    vitg = auto()

    def get_config(self, metric: Optional[Metric] = None) -> Config:
        url = URL_MATRIX[self][metric]
        return {
            Encoder.vits: Config(url=url, features=64, out_channels=[48, 96, 192, 384]),
            Encoder.vitb: Config(
                url=url, features=128, out_channels=[96, 192, 384, 768]
            ),
            Encoder.vitl: Config(
                url=url, features=256, out_channels=[256, 512, 1024, 1024]
            ),
            Encoder.vitg: Config(
                url=url, features=512, out_channels=[1536, 1536, 1536, 1536]
            ),
        }[self]


URL_MATRIX = {
    Encoder.vits: {
        None: "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
        Metric.indoor: "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true",
        Metric.outdoor: "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true",
    },
    Encoder.vitb: {
        None: "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
        Metric.indoor: "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth?download=true",
        Metric.outdoor: "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/main/depth_anything_v2_metric_vkitti_vitb.pth?download=true",
    },
    Encoder.vitl: {
        None: "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
        Metric.indoor: "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true",
        Metric.outdoor: "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true",
    },
    Encoder.vitg: {
        None: "Coming Soon",  # TODO
        Metric.indoor: "Coming Soon",
        Metric.outdoor: "Coming Soon",
    },
}
