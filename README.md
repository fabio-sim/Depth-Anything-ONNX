[![ONNX](https://img.shields.io/badge/ONNX-grey)](https://onnx.ai/)
[![GitHub Repo stars](https://img.shields.io/github/stars/fabio-sim/Depth-Anything-ONNX)](https://github.com/fabio-sim/Depth-Anything-ONNX/stargazers)
[![GitHub all releases](https://img.shields.io/github/downloads/fabio-sim/Depth-Anything-ONNX/total)](https://github.com/fabio-sim/Depth-Anything-ONNX/releases)

# Depth Anything ONNX

Open Neural Network Exchange (ONNX) compatible implementation of [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://github.com/LiheYoung/Depth-Anything) and [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2). Supports PyTorch 2 Export via TorchDynamo.

<p align="center"><img src="assets/demo.gif" autoplay width=90%></p>

<p align="center"><em>Monocular Depth Estimation with Depth Anything V2</em></p>

<p align="center"><img src="assets/sample.png" width=90%></p>

#### Changelog:
- **22 June 2024**: Support Depth Anything V2 and TorchDynamo Export.
- **22 January 2024**: Release.

## 🔥 ONNX Export & Inference

We provide a simple command-line tool [`dynamo.py`](dynamo.py) based on [Typer](https://github.com/tiangolo/typer) to export Depth Anything V2 to ONNX and PyTorch2 programs. Please install the [requirements](/requirements.txt) first.

```shell
$ python dynamo.py --help

 Usage: dynamo.py [OPTIONS] COMMAND [ARGS]...                            
                                                                         
 Depth-Anything Dynamo CLI                                               
                                                                         
╭─ Commands ───────────────────────────────────────────────╮
│ export   Export Depth-Anything V2 using TorchDynamo.     │
│ infer    Depth-Anything V2 inference using ONNXRuntime.  │
│          No dependency on PyTorch.                       │
╰──────────────────────────────────────────────────────────╯
```

If you would like to try out inference right away, you can download ONNX models that have already been exported [here](https://github.com/fabio-sim/Depth-Anything-ONNX/releases).

We observe the following average latencies using the CUDA Execution Provider:

| Device | Encoder | Input Shape | Average Latency (ms) |
| --- | --- | --- | --- |
| RTX4080 12GB | ViT-S | `(1, 3, 518, 518)` | 13.3 |
| RTX4080 12GB | ViT-B | `(1, 3, 518, 518)` | 29.3 |
| RTX4080 12GB | ViT-L | `(1, 3, 518, 518)` | 83.2 |

Relevant framework versions:
```text
CUDA==12.1
cuDNN==8.9.2
onnxruntime-gpu==1.18.0
torch==2.3.1
```

### Export Example

> [!TIP]
> You can view the available options at any time by passing `--help`.

```bash
python dynamo.py export --encoder vitb --output weights/vitb.onnx --opset 18
```

> [!CAUTION]
> The TorchDynamo-based ONNX Exporter is a new beta feature that may undergo breaking changes in the future. Currently, only opset version 18 is supported. Specifying a smaller opset version will fall back to the legacy TorchScript-based Exporter.

### Inference Example

```bash
python dynamo.py infer weights/vitb.onnx -i assets/sacre_coeur1.jpg
```

This function serves as an implementation reference for performing inference with only ONNXRuntime and OpenCV as dependencies.

---

### Legacy
<details>
<summary> V1 </summary>
## 🔥 ONNX Export

Prior to exporting the ONNX models, please install the [requirements](/requirements.txt).

To convert the Depth Anything models to ONNX, run [`export.py`](/export.py). The pretrained weights will be downloaded automatically.

<details>
<summary>Export Example</summary>
<pre>
python export.py --model s
</pre>
</details>

If you would like to try out inference right away, you can download ONNX models that have already been exported [here](https://github.com/fabio-sim/Depth-Anything-ONNX/releases).

## ⚡ ONNX Inference

With ONNX models in hand, one can perform inference on Python using ONNX Runtime. See [`infer.py`](/infer.py).

<details>
<summary>Inference Example</summary>
<pre>
python infer.py --img assets/DSC_0410.JPG --model weights/depth_anything_vits14.onnx --viz
</pre>
</details>


## ⏱️ Inference Time Comparison

<p align="center"><img src="assets/latency.png" alt="Latency Comparison" width=90%>

We report the inference time, or latency, of only the model; that is, the time taken for preprocessing, postprocessing, or copying data between the host & device is not measured. The average inference time is defined as the median over all samples in the [MegaDepth](https://arxiv.org/abs/1804.00607) test dataset. We use the data provided by [LoFTR](https://arxiv.org/abs/2104.00680) [here](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md) - a total of 806 images.

Each image is resized such that its size is 518x518 before being fed into the model. The inference time is then measured for all model variants (S, B, L). See [eval.py](/eval.py) for the measurement code.

All experiments are conducted on an i9-12900HX CPU and RTX4080 12GB GPU with `CUDA==11.8.1`, `torch==2.1.2`, and `onnxruntime==1.16.3`.

### Notes

- Currently, the inference speed is bottlenecked by Conv operations.
- ONNXRuntime performs slightly (20-25%) faster for the ViT-L model variant.

</details>

## Credits
If you use any ideas from the papers or code in this repo, please consider citing the authors of [Depth Anything](https://arxiv.org/abs/2401.10891), [Depth Anything V2](https://arxiv.org/abs/2406.09414) and [DINOv2](https://arxiv.org/abs/2304.07193). Lastly, if the ONNX versions helped you in any way, please also consider starring this repository.

```bibtex
@article{yang2024depth,
      title={Depth Anything V2}, 
      author={Lihe Yang and Bingyi Kang and Zilong Huang and Zhen Zhao and Xiaogang Xu and Jiashi Feng and Hengshuang Zhao},
      year={2024},
      eprint={2406.09414},
      archivePrefix={arXiv},
      primaryClass={id='cs.CV' full_name='Computer Vision and Pattern Recognition' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5.'}
}
```

```bibtex
@article{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      journal={arXiv:2401.10891},
      year={2024}
}
```

```bibtex
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```

