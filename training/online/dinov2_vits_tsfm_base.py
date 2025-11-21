from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence

import fire
import gym
import torch
import torch.nn as nn
from torch import optim

from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig, PPOValue

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    TrainingSettings,
)
from architecture.allenact_preprocessors.dino_preprocessors import (
    DataAugmentationPreprocessor,
    DinoViTPreprocessor,
)
from architecture.models.allenact_transformer_models.allenact_dino_transformer import (
    DinoLLAMATxNavActorCritic,
)
from environment.navigation_sensors import (
    SlowAccurateObjectBBoxSensor,
    TaskNaturalLanguageSpecSensor,
    TaskRelevantObjectBBoxSensorOnlineEval,
    TimeStepSensor,
    TrajectorySensor,
)
from training.online.allenact_trainer import OnPolicyRunnerMixin
from training.online.base import ALL_STRETCH_ACTIONS, BaseConfig, BaseConfigParams
from utils.constants.stretch_initialization_utils import (
    INTEL_CAMERA_HEIGHT,
    INTEL_CAMERA_WIDTH,
)
from utils.type_utils import RewardConfig, THORActions
from utils.wandb_logging import SimpleWandbLogging

from architecture.models.allenact_transformer_models.separate_actor_critic import (
    SafeDinoLLAMATxNavActorCriticSeparate,
)

from environment.vision_sensors import (
    RawNavigationStretchRGBSensor,
    RawManipulationStretchRGBSensor,
)
from environment.manipulation_sensors import AnObjectIsInHand

from training.online.loss.customized_loss import PPOLogGrad, SafePPOLogGrad
from utils.saferl_utils import SafePPOValue, SafeRolloutBlockStorage


@dataclass
class DinoV2ViTSTSFMBaseParams(BaseConfigParams):
    use_data_augmentation: bool = True
    use_text_goal: bool = True
    use_bbox: bool = False
    traj_max_index: int = 2048
    use_traj_indexing: bool = True
    advance_scene_rollout_period: Optional[int] = None
    steps_in_house_before_force_scene_advance: int = 2000
    wandb_project: str = ""
    wandb_entity: str = ""
    collision_penalty: float = -0.00
    lr: float = 2e-5

    # preprocess params
    rgb_height: int = 224
    rgb_width: int = 384

    # training pipeline params
    save_interval: int = 50_000
    metric_accumulate_interval: int = 1_000

    # overwrite the BaseConfigParams tag
    tag: str = "SafeVLA-ObjectNavType-RL-DinoV2-ViTS-TSFM"

    use_sep_critic: bool = True
    full_sensor: bool = True

    il_ckpt_path: Optional[str] = None


class DinoV2ViTSTSFMBase(BaseConfig):

    def __init__(
        self,
        params: DinoV2ViTSTSFMBaseParams,
    ):
        super().__init__(params)
        self.params = params

    def make_sampler_fn(self, **kwargs):
        kwargs["task_args"]["reward_config"] = RewardConfig(
            step_penalty=0.00,
            goal_success_reward=10.0,
            failed_stop_reward=0.0,
            shaping_weight=0.0,
            reached_horizon_reward=0.0,
            positive_only_reward=False,
            failed_action_penalty=self.params.collision_penalty,
        )
        return super().make_sampler_fn(**kwargs)

    def preprocessors(self) -> Sequence[Preprocessor]:
        preprocessors = [
            DataAugmentationPreprocessor(
                rgb_input_uuid="rgb_raw",
                output_uuid="rgb",
                normalize=True,
                mean=DinoViTPreprocessor.DINO_RGB_MEANS,
                stdev=DinoViTPreprocessor.DINO_RGB_STDS,
                height=self.params.rgb_height,
                width=self.params.rgb_width,
                output_channels=3,
                num_steps_to_change=self.params.max_steps,
                use_augmentation=self.params.use_data_augmentation,
            ),
            DinoViTPreprocessor(
                rgb_input_uuid="rgb",
                dino_model_type="dinov2_vits14",  # TODO: Standardize this
                output_uuid="rgb_dinov2",
                class_emb_only=True,
                input_img_height_width=(self.params.rgb_height, self.params.rgb_width),
                chunk_size=64,
                flatten=False,
                normalize=True,
                num_processes=self.params.num_train_processes // len(self.get_devices("train")),
            ),
        ]

        if self.params.full_sensor:
            preprocessors += [
                DataAugmentationPreprocessor(
                    rgb_input_uuid="manipulation_rgb_raw",
                    output_uuid="manipulation_rgb",
                    normalize=True,
                    mean=DinoViTPreprocessor.DINO_RGB_MEANS,
                    stdev=DinoViTPreprocessor.DINO_RGB_STDS,
                    height=224,
                    width=384,
                    output_channels=3,
                    num_steps_to_change=self.params.max_steps,
                    use_augmentation=self.params.use_data_augmentation,
                ),
                DinoViTPreprocessor(
                    rgb_input_uuid="manipulation_rgb",
                    dino_model_type="dinov2_vits14",
                    output_uuid="manipulation_rgb_dinov2",
                    class_emb_only=True,
                    input_img_height_width=(224, 384),
                    chunk_size=64,
                    flatten=False,
                    normalize=True,
                    num_processes=self.params.num_train_processes // len(self.get_devices("train")),
                ),
            ]

        return preprocessors

    @cached_property
    def sensors(self):
        sensors = [
            RawNavigationStretchRGBSensor(
                width=(INTEL_CAMERA_WIDTH - (INTEL_CAMERA_WIDTH % 32)),
                height=INTEL_CAMERA_HEIGHT,
                uuid="rgb_raw",
            ),
            TimeStepSensor(uuid="time_step", max_time_for_random_shift=0),
            TrajectorySensor(uuid="traj_index", max_idx=self.params.traj_max_index),
        ]

        if self.params.use_text_goal:
            sensors += [
                TaskNaturalLanguageSpecSensor(uuid="natural_language_spec"),
            ]

        if self.params.use_bbox:
            sensors += [
                TaskRelevantObjectBBoxSensorOnlineEval(
                    which_camera="nav", uuid="nav_task_relevant_object_bbox"
                ),
                TaskRelevantObjectBBoxSensorOnlineEval(
                    which_camera="nav",
                    uuid="nav_accurate_object_bbox",
                    original_sensor_to_use=SlowAccurateObjectBBoxSensor,
                ),
            ]

        if self.params.full_sensor:
            sensors += [
                RawManipulationStretchRGBSensor(
                    width=(INTEL_CAMERA_WIDTH - (INTEL_CAMERA_WIDTH % 32)),
                    height=INTEL_CAMERA_HEIGHT,
                    uuid="manipulation_rgb_raw",
                ),
                AnObjectIsInHand(uuid="an_object_is_in_hand"),
            ]

        return sensors

    def create_model(self, **kwargs) -> nn.Module:
        goal_sensor_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, TaskNaturalLanguageSpecSensor)),
            None,
        )

        if self.params.use_sep_critic:
            model_init_func = SafeDinoLLAMATxNavActorCriticSeparate
        else:
            model_init_func = DinoLLAMATxNavActorCritic

        if self.params.full_sensor:
            manipulation_rgb_dino_preprocessor_uuid = "manipulation_rgb_dinov2"
            an_object_is_in_hand_uuid = "an_object_is_in_hand"
        else:
            manipulation_rgb_dino_preprocessor_uuid = None
            an_object_is_in_hand_uuid = None

        model = model_init_func(
            action_space=gym.spaces.Discrete(len(ALL_STRETCH_ACTIONS)),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_dino_preprocessor_uuid="rgb_dinov2",
            manipulation_rgb_dino_preprocessor_uuid=manipulation_rgb_dino_preprocessor_uuid,
            an_object_is_in_hand_uuid=an_object_is_in_hand_uuid,
            num_tx_layers=3,
            num_tx_heads=8,
            hidden_size=512,
            goal_dims=512,
            add_prev_actions=True,
            add_prev_action_null_token=True,
            auxiliary_uuids=[],
            max_steps=self.params.max_steps,
            time_step_uuid="time_step",
            initial_tgt_cache_shape=(
                self.params.max_steps,
                (
                    self.params.num_train_processes // len(self.get_devices("train"))
                    if len(self.get_devices("train")) > 0
                    else self.params.num_train_processes
                ),
                512,
            ),
            traj_idx_uuid="traj_index" if self.params.use_traj_indexing else None,
            traj_max_idx=self.params.traj_max_index if self.params.use_traj_indexing else None,
            relevant_object_box_uuid=(
                "nav_task_relevant_object_bbox" if self.params.use_bbox else None
            ),
            accurate_object_box_uuid="nav_accurate_object_bbox" if self.params.use_bbox else None,
            prev_checkpoint=self.params.il_ckpt_path,
        )

        non_nav_action_inds = [
            i
            for i, a in enumerate(ALL_STRETCH_ACTIONS)
            if a
            not in [
                THORActions.move_ahead,
                THORActions.rotate_right,
                THORActions.rotate_left,
                THORActions.move_back,
                THORActions.done,
                THORActions.rotate_right_small,
                THORActions.rotate_left_small,
            ]
        ]

        if not self.params.full_sensor:
            for i in non_nav_action_inds:
                model.actor.linear.bias.data[i] = -999999

        return model

    def training_pipeline(self, **kwargs):
        log_interval_small = (
            self.params.num_train_processes * 128 * 5 if torch.cuda.is_available() else 1
        )
        log_interval_medium = (
            self.params.num_train_processes * 128 * 5 if torch.cuda.is_available() else 1
        )
        log_interval_large = (
            self.params.num_train_processes * 128 * 5 if torch.cuda.is_available() else 1
        )

        batch_steps_0 = int(200000)
        batch_steps_1 = int(800000)
        batch_steps_2 = int(1e9) - batch_steps_1 - batch_steps_0

        NewPPOConfig = dict(
            clip_param=0.1,
            value_loss_coef=0.5,
            entropy_coef=0.0,
            use_clipped_value_loss=False,
            action_loss_schedule=None,
            discrete_critics=False,
            normalize_advantage=False,
        )

        assert (
            self.params.advance_scene_rollout_period is None
        ), "use STEPS_IN_HOUSE_BEFORE_FORCE_SCENE_ADVANCE instead"

        return TrainingPipeline(
            save_interval=self.params.save_interval,
            metric_accumulate_interval=self.params.metric_accumulate_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=self.params.lr)),
            num_mini_batch=1,
            update_repeats=4,
            max_grad_norm=0.5,
            named_losses={
                "ppo_log_loss": SafePPOLogGrad(**NewPPOConfig),
                "ppo_value_loss": PPOValue(clip_param=0.1, use_clipped_value_loss=False),
                "safe_ppo_value_loss": SafePPOValue(clip_param=0.1, use_clipped_value_loss=False),
            },
            named_storages={
                "onpolicy": Builder(SafeRolloutBlockStorage),
            },
            gamma=0.99,
            use_gae=True,
            gae_lambda=0.95,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_value_loss", "safe_ppo_value_loss"],
                    max_stage_steps=batch_steps_0,
                    training_settings=TrainingSettings(
                        num_steps=128,
                        metric_accumulate_interval=log_interval_small,
                        advance_scene_rollout_period=self.params.steps_in_house_before_force_scene_advance
                        // 128,
                    ),
                ),
                PipelineStage(
                    loss_names=["ppo_log_loss"],
                    max_stage_steps=batch_steps_1,
                    training_settings=TrainingSettings(
                        num_steps=128,
                        metric_accumulate_interval=log_interval_medium,
                        advance_scene_rollout_period=self.params.steps_in_house_before_force_scene_advance
                        // 128,
                    ),
                ),
                PipelineStage(
                    loss_names=["ppo_log_loss"],
                    max_stage_steps=batch_steps_2,
                    training_settings=TrainingSettings(
                        num_steps=128,
                        metric_accumulate_interval=log_interval_large,
                        advance_scene_rollout_period=self.params.steps_in_house_before_force_scene_advance
                        // 128,
                    ),
                ),
            ],
        )

    def wandb_logging_callback(self) -> SimpleWandbLogging:
        assert self.params.wandb_entity is not None and self.params.wandb_project is not None, (
            "Entity and project must be set to use wandb logging."
            " Set these values when specifying the --config_kwargs when running the experiment."
        )
        return SimpleWandbLogging(
            project=self.params.wandb_project, entity=self.params.wandb_entity
        )


@dataclass
class DinoV2ViTSTSFMBaseRunner(DinoV2ViTSTSFMBaseParams, OnPolicyRunnerMixin):
    def get_config(self):
        return DinoV2ViTSTSFMBase(self)


if __name__ == "__main__":
    fire.Fire(DinoV2ViTSTSFMBaseRunner)
