import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "framework",
        type=str,
        choices=["torch", "ort"],
        help="The framework to measure inference time. Options are 'torch' for PyTorch and 'ort' for ONNXRuntime.",
    )
    parser.add_argument(
        "--megadepth_path",
        type=Path,
        default=Path("megadepth_test_1500"),
        required=False,
        help="Path to the root of the MegaDepth dataset.",
    )

    # PyTorch-specific args
    parser.add_argument(
        "--model",
        type=str,
        choices=["s", "b", "l"],
        required=False,
        help="Model size variant. Available options: 's', 'b', 'l'.",
    )

    # ONNXRuntime-specific args
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        required=False,
        help="Path to ONNX model.",
    )
    return parser.parse_args()


def get_megadepth_images(path: Path):
    sort_key = lambda p: int(p.stem.split("_")[0])
    images = sorted(
        list((path / "Undistorted_SfM/0015/images").glob("*.jpg")), key=sort_key
    ) + sorted(list((path / "Undistorted_SfM/0022/images").glob("*.jpg")), key=sort_key)
    return images


def create_models(framework: str, model=None, onnx_path=None):
    if framework == "torch":
        device = torch.device("cuda")
        assert model is not None, "Model size variant must be specified."

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
    elif framework == "ort":
        sess_opts = ort.SessionOptions()
        # sess_opts.intra_op_num_threads = 1
        # sess_opts.enable_profiling = True
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        assert onnx_path is not None, "ONNX model path must be specified."

        depth_anything = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=providers,
        )

    return depth_anything


def measure_inference(framework: str, depth_anything, image) -> float:
    if framework == "torch":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            result = depth_anything(image)
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end)
    elif framework == "ort":
        inputs = {"image": image}
        outputs = ["depth"]

        # Prepare IO-Bindings
        binding = depth_anything.io_binding()

        for name, arr in inputs.items():
            binding.bind_cpu_input(name, arr)

        for name in outputs:
            binding.bind_output(name, "cuda")

        # Measure only matching time
        start = time.perf_counter()
        result = depth_anything.run_with_iobinding(binding)
        end = time.perf_counter()

        return (end - start) * 1000


def evaluate(
    framework, megadepth_path=Path("megadepth_test_1500"), model=None, onnx_path=None
):
    images = get_megadepth_images(megadepth_path)

    depth_anything = create_models(
        framework=framework, model=model, onnx_path=onnx_path
    )

    # Warmup
    for img in images[:10]:
        image, _ = load_image(str(img))

        if framework == "torch":
            image = torch.from_numpy(image).cuda()
        elif framework == "ort":
            pass

        _ = measure_inference(framework, depth_anything, image)

    # Measure
    timings = []
    for img in tqdm(images[10:]):
        image, _ = load_image(str(img))

        if framework == "torch":
            image = torch.from_numpy(image).cuda()
        elif framework == "ort":
            pass

        inference_time = measure_inference(framework, depth_anything, image)
        timings.append(inference_time)

    # Results
    timings = np.array(timings)
    print(f"Mean inference time: {timings.mean():.2f} +/- {timings.std():.2f} ms")
    print(f"Median inference time: {np.median(timings):.2f} ms")


if __name__ == "__main__":
    args = parse_args()
    evaluate(**vars(args))
