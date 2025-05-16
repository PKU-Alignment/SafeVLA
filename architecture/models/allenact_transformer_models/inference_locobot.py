import os

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


class InferenceAgentLocoBot(InferenceAgent):
    img_encoder_rgb_mean = attr.ib()
    img_encoder_rgb_std = attr.ib()
    greedy_sampling: bool = attr.ib()
    test_augmentation: bool = attr.ib()
    resize_and_normalize_rgb = attr.ib()
    num_evaluated_traj = 0
    cfg: dict = None
    target_object_types: list = None

    @classmethod
    def build_agent(
        cls,
        exp_config_type,
        cfg,
        device,
        greedy_sampling,
        ckpt_path,
    ):
        exp_config = exp_config_type(cfg)
        agent = cls.from_experiment_config(
            exp_config=exp_config,
            mode="test",
            device=device,
        )
        agent.img_encoder_rgb_mean = cfg.model.rgb_means
        agent.img_encoder_rgb_std = cfg.model.rgb_stds
        agent.greedy_sampling = greedy_sampling
        agent.resize_and_normalize_rgb = tensor_image_preprocessor(
            size=(224, 224),
            data_augmentation=False,
            mean=cfg.model.rgb_means,
            std=cfg.model.rgb_stds,
        )

        agent.actor_critic.load_state_dict(
            torch.load(ckpt_path, map_location="cpu" if not torch.cuda.is_available() else "cuda")[
                "model_state_dict"
            ]
        )

        agent.steps_before_rollout_refresh = 10000

        agent.cfg = cfg
        agent.target_object_types = [obj.lower() for obj in cfg.target_object_types]
        agent.reset()
        return agent

    def reset(self):
        if self.has_initialized:
            self.rollout_storage.after_updates()
        self.steps_taken_in_task = 0
        self.num_evaluated_traj += 1
        self.memory = None

    def get_action_list(self):
        return self.cfg.mdp.actions

    def resize_and_normalize(self, frame):
        frame = self.resize_and_normalize_rgb(torch.Tensor(frame).permute(2, 0, 1)).permute(1, 2, 0)
        return frame

    def get_action(self, frame, goal_spec):
        if goal_spec.split(" ")[-1] == "bottle":
            target_id = self.target_object_types.index("spraybottle")
        elif goal_spec.split(" ")[-1] == "clock":
            target_id = self.target_object_types.index("alarmclock")
        elif goal_spec.split(" ")[-1] == "can":
            target_id = self.target_object_types.index("garbagecan")
        else:
            target_id = self.target_object_types.index(goal_spec.split(" ")[-1])
        observations = {
            "rgb_lowres": self.resize_and_normalize(frame["raw_navigation_camera"]),
            "goal_object_type_ind": target_id,
            "time_step": self.steps_taken_in_task,
            "traj_index": self.num_evaluated_traj,
        }
        if "manipulation_rgb_raw" in frame.keys():
            observations["manipulation_rgb_raw"] = frame["manipulation_rgb_raw"]
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
            if len(frame["nav_task_relevant_object_bbox"]) == 5:
                pass
            elif len(frame["nav_task_relevant_object_bbox"]) == 10:
                frame["nav_task_relevant_object_bbox"] = frame["nav_task_relevant_object_bbox"][:5]
            else:
                raise NotImplementedError
            observations["nav_task_relevant_object_bbox"] = frame["nav_task_relevant_object_bbox"]
        return self.act(observations)

    def act(self, observations: ObservationType):
        obs_batch = batch_observations([observations], device=self.device)
        if self.sensor_preprocessor_graph is not None:
            obs_batch = self.sensor_preprocessor_graph.get_observations(obs_batch)

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

        if self.greedy_sampling:
            action_idx = su.action_list(self.actor_critic.action_space, action_greedy)[0]
        else:
            action_idx = su.action_list(self.actor_critic.action_space, self.last_action_flat)[0]

        action_str = 0
        if action_idx in [0, 1, 2, 3, 4, 5]:
            if action_idx == 0:
                action_str = ALL_STRETCH_ACTIONS[0]
            elif action_idx == 1:
                action_str = ALL_STRETCH_ACTIONS[2]
            elif action_idx == 2:
                action_str = ALL_STRETCH_ACTIONS[1]
            elif action_idx == 3:
                action_str = ALL_STRETCH_ACTIONS[4]
            elif action_idx == 4:
                action_str = ALL_STRETCH_ACTIONS[6]
            elif action_idx == 5:
                action_str = ALL_STRETCH_ACTIONS[7]
        else:
            raise NotImplementedError("NO!!! YOU SHOULD NOT BE HERE!!!")

        return action_str, actor_critic_output.distributions.probs[0][0]
