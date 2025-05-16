import torch
from allenact.base_abstractions.misc import (
    ObservationType,
    ActorCriticOutput,
    DistributionType,
    Memory,
)
from allenact.utils.inference import InferenceAgent
from allenact.utils.tensor_utils import batch_observations
from allenact.utils import spaces_utils as su
from typing import Optional, cast, Tuple
from utils.constants.stretch_initialization_utils import ALL_STRETCH_ACTIONS
from utils.string_utils import convert_string_to_byte
import numpy as np
import attr
from PIL import Image
from architecture.models.transformer_models.preprocessors import tensor_image_preprocessor


class TrainingAgentVIDA(InferenceAgent):
    img_encoder_rgb_mean = attr.ib()
    img_encoder_rgb_std = attr.ib()
    augmentations = attr.ib()
    num_evaluated_traj = 0

    @classmethod
    def build_agent(
        cls,
        exp_config_type,
        device,
        img_encoder_rgb_mean,
        img_encoder_rgb_std,
        ckpt_path=None,
    ):
        exp_config = exp_config_type(num_train_processes=1, train_gpu_ids=[device])
        agent = cls.from_experiment_config(
            exp_config=exp_config,
            mode="test",
            device=device,
        )
        agent.img_encoder_rgb_mean = img_encoder_rgb_mean
        agent.img_encoder_rgb_std = img_encoder_rgb_std
        agent.augmentations = tensor_image_preprocessor(
            size=(256, 256),
            data_augmentation=True,
            augmentation_version="v2",
            mean=img_encoder_rgb_mean,
            std=img_encoder_rgb_std,
        )

        if ckpt_path is not None:
            agent.actor_critic.load_state_dict(
                torch.load(
                    ckpt_path, map_location="cpu" if not torch.cuda.is_available() else "cuda"
                )["model_state_dict"]
            )

        agent.steps_before_rollout_refresh = 10000

        agent.reset()
        return agent

    def reset(self):
        if self.has_initialized:
            self.rollout_storage.after_updates()
        self.steps_taken_in_task = 0
        self.num_evaluated_traj += 1
        self.memory = None

    def normalize_img(self, frame):
        frame -= (
            torch.from_numpy(np.array(self.img_encoder_rgb_mean))
            .to(device=self.device)
            .float()
            .view(1, 1, 1, 3)
        )
        frame /= (
            torch.from_numpy(np.array(self.img_encoder_rgb_std))
            .to(device=self.device)
            .float()
            .view(1, 1, 1, 3)
        )
        return frame

    def get_action_list(self):
        return ALL_STRETCH_ACTIONS

    def get_action(self, frame, goal_spec):
        observations = {
            "rgb_raw": frame["raw_navigation_camera"],
            "natural_language_spec": convert_string_to_byte(goal_spec, 1000),
            "time_step": self.steps_taken_in_task,
            "traj_index": self.num_evaluated_traj,
        }
        if "manipulation_rgb_raw" in frame.keys():
            observations["manipulation_rgb_raw"] = frame["raw_navigation_camera"]
        if "an_object_is_in_hand" in frame.keys():
            observations["an_object_is_in_hand"] = frame["an_object_is_in_hand"]
        if "relative_arm_location_metadata" in frame.keys():
            full_pose = frame["relative_arm_location_metadata"]
            full_pose[-1] = full_pose[-1] * np.pi / 180
            full_pose[-1] = (full_pose[-1] + np.pi) % (2 * np.pi) - np.pi
            observations["relative_arm_location_metadata"] = full_pose
        if "nav_accurate_object_bbox" in frame.keys():
            observations["nav_accurate_object_bbox"] = frame["nav_accurate_object_bbox"]
        if "nav_task_relevant_object_bbox" in frame.keys():
            observations["nav_task_relevant_object_bbox"] = frame["nav_task_relevant_object_bbox"]
        return self.act(observations)

    def act(self, observations: ObservationType):
        obs_batch = batch_observations([observations], device=self.device)
        if self.sensor_preprocessor_graph is not None:
            if "graph" in self.sensor_preprocessor_graph.compute_order:
                self.sensor_preprocessor_graph.compute_order.pop(
                    self.sensor_preprocessor_graph.compute_order.index("graph")
                )
            obs_batch = self.sensor_preprocessor_graph.get_observations(obs_batch)
        if "rgb_dino_vit" in obs_batch.keys():
            obs_batch["rgb_dino_vit"] = (
                obs_batch["rgb_dino_vit"].flatten(start_dim=2).permute(0, 2, 1)
            )
        if "graph" in obs_batch.keys():
            obs_batch.pop("graph")

        if self.steps_taken_in_task == 0:
            self.has_initialized = True
            self.rollout_storage.initialize(
                observations=obs_batch,
                num_samplers=1,
                recurrent_memory_specification=self.actor_critic.recurrent_memory_specification,
                action_space=self.actor_critic.action_space,
            )
            self.rollout_storage.after_updates()
        else:
            dummy_val = torch.zeros((1, 1), device=self.device)  # Unused dummy value
            self.rollout_storage.add(
                observations=obs_batch,
                memory=self.memory,
                actions=self.last_action_flat[0],
                action_log_probs=dummy_val,
                value_preds=dummy_val,
                rewards=dummy_val,
                masks=torch.ones(
                    (1, 1), device=self.device
                ),  # Always == 1 as we're in a single task until `reset`
            )

        agent_input = self.rollout_storage.agent_input_for_next_step()

        actor_critic_output, self.memory = cast(
            Tuple[ActorCriticOutput[DistributionType], Optional[Memory]],
            self.actor_critic(**agent_input),
        )

        action = actor_critic_output.distributions.sample()

        action_greedy = actor_critic_output.distributions.mode()

        # NOTE: Last action flat is always stochastic
        self.last_action_flat = su.flatten(self.actor_critic.action_space, action)

        self.steps_taken_in_task += 1

        if self.steps_taken_in_task % self.steps_before_rollout_refresh == 0:
            self.rollout_storage.after_updates()

        action_str = self.get_action_list()[
            su.action_list(self.actor_critic.action_space, self.last_action_flat)[0]
        ]

        return action_str, actor_critic_output.distributions.probs[0][0]

    def preprocess(self, observations: ObservationType):
        obs_batch = batch_observations([observations], device=self.device)
        if self.sensor_preprocessor_graph is not None:
            if "graph" in self.sensor_preprocessor_graph.compute_order:
                self.sensor_preprocessor_graph.compute_order.pop(
                    self.sensor_preprocessor_graph.compute_order.index("graph")
                )
            obs_batch = self.sensor_preprocessor_graph.get_observations(obs_batch)
        if "rgb_dino_vit" in obs_batch.keys():
            obs_batch["rgb_dino_vit"] = (
                obs_batch["rgb_dino_vit"].flatten(start_dim=2).permute(0, 2, 1)
            )
        if "graph" in obs_batch.keys():
            obs_batch.pop("graph")
        return obs_batch

    def forward(self, observations: ObservationType):
        xxx = 0
        return self.act(observations)
