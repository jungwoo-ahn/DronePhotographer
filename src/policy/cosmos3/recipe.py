"""Teach `cosmos_framework` about the camera policy, for inference.

The checkpoint is a Cosmos3-Nano action-SFT net fine-tuned on a 10D camera action. The
framework ships neither that action type nor that experiment, so they are registered from
here, which leaves the vendored checkout in `repos/` unpatched. Import this before the
framework resolves its config.

The model settings are the ones the checkpoint trained with. Everything about training
(optimizer, callbacks, dataloaders) is left out: nothing reads it here, and the training
callbacks drag in modules this repo does not have.
"""

from __future__ import annotations

import copy

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.domain_utils import (
    EMBODIMENT_TO_DOMAIN_ID,
    EMBODIMENT_TO_RAW_ACTION_DIM,
)
from cosmos_framework.utils.lazy_config import LazyDict
from hydra.core.config_store import ConfigStore

EXPERIMENT = "action_policy_camera_nano"
DOMAIN_NAME = "camera_pose_shoot"
CAMERA_ACTION_DIM = 10                 # 3 pos + 6 rot6d + 1 shoot
CHUNK_LENGTH = 8

# The domain id has to stay camera_pose's: `DomainAwareLinear` keeps one weight matrix per
# id, and the camera_pose row is the one that was fine-tuned.
EMBODIMENT_TO_DOMAIN_ID.setdefault(DOMAIN_NAME, EMBODIMENT_TO_DOMAIN_ID["camera_pose"])
EMBODIMENT_TO_RAW_ACTION_DIM.setdefault(DOMAIN_NAME, CAMERA_ACTION_DIM)


def _model_config() -> dict:
    cfg = copy.deepcopy(NANO_MODEL_CONFIG)       # action_gen=True, max_action_dim=64
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
    cfg["rectified_flow_training_config"]["image_loss_scale"] = None
    # Has to contain the video window, which is one frame longer than the action chunk.
    cfg["tokenizer"]["encode_exact_durations"] = [CHUNK_LENGTH + 1, 61, 73]
    return cfg


action_policy_camera_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {"override /callbacks": ["basic"]},
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(project="cosmos3", group="action_sft", name=EXPERIMENT, wandb_mode="disabled"),
        model=dict(config=_model_config()),
        trainer=dict(distributed_parallelism="fsdp", run_validation=False),
        checkpoint=dict(load_training_state=False, strict_resume=False),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

ConfigStore.instance().store(group="experiment", package="_global_", name=EXPERIMENT,
                             node=action_policy_camera_nano)
