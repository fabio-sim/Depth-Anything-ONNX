import argparse

import cv2
import numpy as np
import onnxruntime as ort

from depth_anything.util.transform import load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img",
        type=str,
        required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model.",
    )
    parser.add_argument(
        "--viz", action="store_true", help="Whether to visualize the results."
    )
    return parser.parse_args()


def infer(img: str, model: str, viz: bool = False):
    image, (orig_h, orig_w) = load_image(img)

    session = ort.InferenceSession(
        model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    depth = session.run(None, {"image": image})[0]

    depth = cv2.resize(depth[0, 0], (orig_w, orig_h))
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    # Visualization
    if viz:
        margin_width = 50
        caption_height = 60
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        split_region = np.ones((orig_h, margin_width, 3), dtype=np.uint8) * 255
        combined_results = cv2.hconcat([cv2.imread(img), split_region, depth_color])

        caption_space = (
            np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8)
            * 255
        )
        captions = ["Raw image", "Depth Anything"]
        segment_width = orig_w + margin_width
        for i, caption in enumerate(captions):
            # Calculate text size
            text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

            # Calculate x-coordinate to center the text
            text_x = int((segment_width * i) + (orig_w - text_size[0]) / 2)

            # Add text caption
            cv2.putText(
                caption_space,
                caption,
                (text_x, 40),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

        final_result = cv2.vconcat([caption_space, combined_results])

        cv2.imshow("depth", final_result)
        cv2.waitKey(0)

    return depth


if __name__ == "__main__":
    args = parse_args()
    infer(**vars(args))
