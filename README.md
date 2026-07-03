<div align="center">
  <h1>OpenClaw-RL-Ascend</h1>
  <p><b>OpenClaw-RL adapted for Ascend NPUs — train a personalized agent simply by talking to it.</b></p>
  <p>
    <img src="https://img.shields.io/badge/Supported-Ascend NPU-red?style=flat-square" alt="Ascend NPU Supported" />
    <img src="https://img.shields.io/badge/Training-Agentic RL-yellow?style=flat-square" alt="Agentic RL Training" />
    <img src="https://img.shields.io/badge/License-Apache_2.0-green?style=flat-square" alt="License Apache 2.0" />
  </p>
</div>

---

OpenClaw-RL-Ascend is an adaptation of the open-source project [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL), optimized for running on **Ascend NPUs**. Currently, **openclaw-combine** (Binary RL + OPD) training has been fully adapted. Other training modes are being adapted continuously.

> See the original upstream project README: [OpenClaw-RL](./README_opensource.md)

---

## Ascend Adaptations

### 1. slime-ascend Adaptation

The `slime-ascend` directory in this repository is a **git submodule** that adapts the original [slime](https://github.com/THUDM/slime) RL training framework for Ascend NPUs. It incorporates the same modifications that the upstream OpenClaw-RL project applies to its own `slime` submodule, ported to the Ascend ecosystem:

- Custom loss functions, rollout logic, and API server adapters ported to run on Ascend hardware
- Ascend-specific environment variables and HCCL communication configurations
- NPU-aware memory management and device placement

All method-specific code in this repository — including `openclaw-combine` — relies on `slime-ascend` as the training backend.

### 2. Other Changes from Upstream

In addition to the Ascend NPU adaptation, the following enhancements have been made on top of the original OpenClaw-RL codebase:

- **Remote PRM extension** — added support for offloading judge/eval scoring to a remote OpenAI-compatible API/API key, allowing stronger models to be used for evaluation
- **Detailed PRM logging** — added step-by-step PRM recording and expanded training debug logs for better observability
- **Additional model support** — added training adaptation for more models (e.g., GLM-4.7-Flash)

---

## Quick Start

### 1. Hardware Requirements

An **Ascend A3** machine with **16 NPUs** is recommended.

### 2. Build the slime-ascend Base Image

OpenClaw-RL-Ascend depends on the [slime-ascend](https://github.com/THUDM/slime) training framework. Build the slime-ascend base image first using the official v0.2.2 Dockerfile:

> [Dockerfile.a3.ubuntu22.04.cann850.latest](https://gitcode.com/Ascend/slime-ascend/blob/v0.2.2/docker/npu_docker/v0.2.2/Dockerfile.a3.ubuntu22.04.cann850.latest)

### 3. Build the OpenClaw-RL-Ascend Image

After the slime-ascend image is ready, clone the OpenClaw-RL-Ascend repository and build the runtime image on top of the **slime-ascend v0.2.2** base image. The recommended Docker version is **26.1.4**.

```Dockerfile
FROM slime-ascend:8.5.0-a3-ubuntu22.04.cann850.latest

COPY ./OpenClaw-RL-Ascend /workspace/OpenClaw-RL

RUN cd "$(pip show deep-ep | grep -E 'Location:' | awk '{print $2}')" && \
    ln -sf deep_ep/deep_ep_cpp*.so && cd -
RUN pip install fastapi==0.123.10 && \
    pip install fastapi-cli==0.0.24 && \
    pip install fastapi-cloud-cli==0.17.0

RUN cd /workspace && \
    rm -rf slime-ascend && \
    cd /workspace/OpenClaw-RL/slime-ascend && \
    pip install -e .

RUN git clone https://gitcode.com/cann/cann-recipes-infer.git && \
    cd cann-recipes-infer/ && \
    git reset --hard bc80902bfe64a317cb0dc4b1aba90f958bc3a6fb && \
    cd ops/ascendc && \
    bash build.sh && \
    ./output/CANN-custom_ops-*.run && \
    cd torch_ops_extension && \
    bash build_and_install.sh && \
    cd /workspace

RUN pip list

CMD ["/bin/bash"]
```

Once the image is built, start and enter the container.

### 4. Start Training

In the container, use one of the following ready-to-run scripts to launch openclaw-combine training:

- [run_glm_4.7_flash_openclaw_combine_npu.sh](./openclaw-combine/run_glm_4.7_flash_openclaw_combine_npu.sh) — GLM-4.7-Flash on Ascend NPU
- [run_qwen3_4b_openclaw_combine_npu.sh](./openclaw-combine/run_qwen3_4b_openclaw_combine_npu.sh) — Qwen3-4B on Ascend NPU

For parameter usage and adjustments, refer to the comments inside each script.

### 5. Connect to OpenClaw

Once the script starts successfully, the OpenClaw-RL-Ascend RL service is up and running on the default port **30000**, exposed as an OpenAI-compatible inference API. You can connect your OpenClaw to this endpoint. OpenClaw-RL-Ascend will serve inference requests while simultaneously collecting on-policy training data and asynchronously updating the model weights in the background.

---

## Dependencies

The following key dependencies are required. Other dependencies follow the configuration of slime-ascend.

| Dependency | Version | Commit |
|---|---|---|
| [slime-ascend](https://github.com/THUDM/slime) | v0.2.2 | `21328a8f` |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | main | `3714d81d` |
| [mbridge](https://github.com/ISEEKYAN/mbridge) | main | `89eb1088` |
| [Megatron-Bridge](https://github.com/NVIDIA/Megatron-Bridge) | dev_rl | `35b4ebf` |
| [MindSpeed](https://gitee.com/ascend/MindSpeed) | master | `fc63de5c` |
| [sglang](https://github.com/sgl-project/sglang) | v0.5.8 | `0189f41` |

For additional dependencies, refer to the configuration in slime-ascend.

---

## Acknowledgements

This project is based on [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL). We sincerely thank the original authors for their outstanding work.

**Original OpenClaw-RL authors:** Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, Ling Yang

```bibtex
@article{wang2026openclawrl,
  title={OpenClaw-RL: Train Any Agent Simply by Talking},
  author={Wang, Yinjie and Chen, Xuyang and Jin, Xiaolong and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2603.10165},
  year={2026}
}

@article{wang2026rlanything,
  title={RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System},
  author={Wang, Yinjie and Xie, Tianbao and Shen, Ke and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2602.02488},
  year={2026}
}
```

---

## License

This project is released under the [Apache 2.0 License](./LICENSE), following the original OpenClaw-RL project.
