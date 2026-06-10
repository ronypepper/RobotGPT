"""
This script defines classes/configs for interfacing/training openpi models with the Isaac Lab environments of the
RobotGPT repository, as well as with datasets created using these environments.

Three main components are defined:
- *Input / *Output classes that define the data mapping from the Isaac Lab environment to the model and vice versa.
- A *DataConfig class that defines how to process raw data from datasets for training.
- Several TrainConfig class instances that define fine-tuning hyperparameters, data config, and weight loader.


Parts of this script are based on contents of the "src/openpi/policies/libero_policy.py" and
"src/openpi/training/config.py" scripts of the openpi repository.

Modifications: Copyright (c) 2026 ronypepper.

License: Apache 2.0
"""

from dataclasses import MISSING
import pathlib

import einops
import numpy as np
from typing_extensions import override

from openpi import transforms
from openpi.models import model as _model
import openpi.models.pi0_config as pi0_config
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory
from openpi.training.config import TrainConfig
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


def make_franka_example() -> dict:
    """Creates a random input example."""
    return {
        "observation/state": np.random.rand(8),
        "observation/table_img": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_img": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaSingleArmInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        table_image = _parse_image(data["observation/table_img"])
        wrist_image = _parse_image(data["observation/wrist_img"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/joint_pos"],
            "image": {
                "base_0_rgb": table_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(table_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaSingleArmOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 8 actions.
        return {"actions": np.asarray(data["actions"][:, :8])}


@dataclasses.dataclass(frozen=True)
class FrankaSingleArmDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    """

    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/table_img": "table_img",
                        "observation/wrist_img": "wrist_img",
                        "observation/joint_pos": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        data_transforms = _transforms.Group(
            inputs=[FrankaSingleArmInputs(model_type=model_config.model_type)],
            outputs=[FrankaSingleArmOutputs()],
        )

        # Pi0 models are trained on delta actions (relative to the first state in each action chunk), except for the gripper.
        # Data loader returns absolute joint position actions -- convert to delta actions for training.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# These train configs define the hyperparameters for fine-tuning the base model on a dataset.
franka_single_arm_configs = [
    TrainConfig(
        name="robotgpt_franka_single_arm_pi05_lora",  # low_mem_finetune
        model=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ),
        data=FrankaSingleArmDataConfig(
            repo_id=MISSING,
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=5_000,
        save_interval=500,
        keep_period=500,
        # The freeze filter defines which parameters should be frozen during training.
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        batch_size=4,
        num_workers=4
    ),
    # This config is just for viewing the pi05 base model's performance on a task as is - not for finetuning.
    TrainConfig(
        name="robotgpt_franka_single_arm_pi05_base",
        model=pi0_config.Pi0Config(
            pi05=True,
        ),
        data=FrankaSingleArmDataConfig(
            repo_id=MISSING,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="franka",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),
]
