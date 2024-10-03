from modal import App, Image, Volume, Secret
import os

APP_NAME = "uk-finetuning"

# Axolotl image hash corresponding to 0.4.0 release (2024-02-14)
AXOLOTL_REGISTRY_SHA = (
    "8ec2116dd36ecb9fb23702278ac612f27c1d4309eca86ad0afd3a3fe4a80ad5b"
)


axolotl_image = (
    Image.from_registry(f"winglian/axolotl@sha256:{AXOLOTL_REGISTRY_SHA}")
    .pip_install(
        "huggingface_hub==0.20.3",
        "hf-transfer==0.1.5",
        "wandb==0.16.3",
        "fastapi==0.110.0",
        "pydantic==2.6.3",
    )
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
        )
    )
)

vllm_image = Image.from_registry(
    "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
).pip_install(
    "vllm==0.2.6",
    "torch==2.1.2",
)

app = App(APP_NAME, secrets=[Secret.from_name("my-wandb-secret"), Secret.from_name("huggingface")])

# Volumes for pre-trained models and training runs.
pretrained_volume = Volume.from_name("uk-pretrained-vol", create_if_missing=True)
runs_volume = Volume.from_name("uk-runs-vol", create_if_missing=True)
VOLUME_CONFIG: dict[str | os.PathLike, Volume] = {"/pretrained": pretrained_volume, "/runs": runs_volume}
