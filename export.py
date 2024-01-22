import argparse

import torch
from onnx import load_model, save_model
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["s", "b", "l"],
        required=True,
        help="Model size variant. Available options: 's', 'b', 'l'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        required=False,
        help="Path to save the ONNX model.",
    )

    return parser.parse_args()


def export_onnx(model: str, output: str = None):
    # Handle args
    if output is None:
        output = f"weights/depth_anything_vit{model}14.onnx"

    # Device for tracing (use whichever has enough free memory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample image for tracing (dimensions don't matter)
    image, _ = load_image("assets/sacre_coeur1.jpg")
    image = torch.from_numpy(image).to(device)

    # Load model params
    if model == "s":
        depth_anything = DPT_DINOv2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384]
        )
    elif model == "b":
        depth_anything = DPT_DINOv2(
            encoder="vitb", features=128, out_channels=[96, 192, 384, 768]
        )
    else:  # model == "l"
        depth_anything = DPT_DINOv2(
            encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]
        )

    depth_anything.to(device).load_state_dict(
        torch.hub.load_state_dict_from_url(
            f"https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vit{model}14.pth",
            map_location="cpu",
        ),
        strict=True,
    )
    depth_anything.eval()

    torch.onnx.export(
        depth_anything,
        image,
        output,
        input_names=["image"],
        output_names=["depth"],
        opset_version=17,
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            "depth": {2: "height", 3: "width"},
        },
    )

    save_model(
        SymbolicShapeInference.infer_shapes(load_model(output), auto_merge=True),
        output,
    )


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
