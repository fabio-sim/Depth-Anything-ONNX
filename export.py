import argparse
import subprocess

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
        required=False,
        help="Model size variant. Available options: 's', 'b', 'l'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        required=False,
        help="Path to save the ONNX model.",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        required=False,
        help="Precision for the model. Available options: 'float32', 'float16'.",
    )

    parser.add_argument(
        "--slim",
        action="store_true",
        help="Whether to slim the model using ONNXSlim.",
    )

    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Whether to export all models. With all precisions and with slimming, if enabled.",
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=19,
        required=False,
        help="ONNX opset version.",
    )

    return parser.parse_args()


def load_depth_anything(model, device, precision="float32"):
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

    if precision == "float16":
        return depth_anything.half()
    else:
        return depth_anything


def export_onnx(
    depth_anything,
    image,
    output: str,
    opset: int = 19,
):
    torch.onnx.export(
        depth_anything,
        image,
        output,
        input_names=["image"],
        output_names=["depth"],
        opset_version=opset,
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            "depth": {2: "height", 3: "width"},
        },
    )

    save_model(
        SymbolicShapeInference.infer_shapes(load_model(output), auto_merge=True),
        output,
    )


def slim_model(model: str):
    output = model.replace(".onnx", "_slim.onnx")
    try:
        subprocess.run(f"python -m onnxslim {model} {output}", stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except Exception as e:
        print(f"Failed to slim model: {e}")
        return


def main(
    model: str,
    output: str = None,
    export_all: bool = False,
    slim: bool = False,
    precision: str = "float32",
    opset: int = 19,
):

    # Device for tracing (use whichever has enough free memory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample image for tracing (dimensions don't matter)
    image, _ = load_image("assets/sacre_coeur1.jpg")
    image = torch.from_numpy(image).to(device)

    # Export all models with all precision, quick and dirty
    if export_all:
        print("Exporting all models with all precisions... This may take a while.")
        for model in ["s", "b", "l"]:
            for precision in ["float32", "float16"]:
                print(f"Exporting model: {model} with precision: {precision}")
                output = f"weights/depth_anything_vit{model}14_{precision}.onnx"
                
                image = image.float() if precision == "float32" else image.half()
                depth_anything = load_depth_anything(model, device, precision)
                depth_anything = export_onnx(depth_anything, image, output, opset)
                if slim:
                    slimModel(output)

                print(f"Exported model: {model} with precision: {precision}")

        print("All models exported.")

    elif model is not None:
        # Handle args
        if output is None:
            output = f"weights/depth_anything_vit{model}14_{precision}.onnx"
        if precision == "float16":
            image = image.half()
        depth_anything = loadModel(model, device, precision)
        depth_anything = export_onnx(depth_anything, image, output, opset)
        if slim:
            slim_model(output)
    else:
        print("No model specified.")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
