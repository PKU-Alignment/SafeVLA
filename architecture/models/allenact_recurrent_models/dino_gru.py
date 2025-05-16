from typing import Optional, List, Dict, cast, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.embodiedai.models.visual_nav_models import (
    VisualNavActorCritic,
    FusionType,
)
from gym.spaces import Dict as SpaceDict
from transformers import T5EncoderModel, AutoTokenizer

from utils.string_utils import convert_byte_to_string


class DinoTensorNavActorCritic(VisualNavActorCritic):
    def __init__(
        # base params
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
        add_prev_actions=False,
        add_prev_action_null_token=False,
        action_embed_size=6,
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
        # custom params
        rgb_dino_preprocessor_uuid: Optional[str] = None,
        goal_dims: int = 32,
        dino_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        **kwargs,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
            **kwargs,
        )

        if rgb_dino_preprocessor_uuid is not None:
            dino_preprocessor_uuid = rgb_dino_preprocessor_uuid
            self.goal_visual_encoder = DinoTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                dino_preprocessor_uuid,
                goal_dims,
                dino_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )

        self.create_state_encoders(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            action_embed_size=action_embed_size,
        )

        self.train()

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.goal_visual_encoder(observations)


class DinoTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        dino_preprocessor_uuid: str,
        goal_embed_dims: int = 32,
        dino_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.dino_uuid = dino_preprocessor_uuid
        self.goal_embed_dims = goal_embed_dims
        self.dino_hid_out_dims = dino_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.goal_space = observation_spaces.spaces[self.goal_uuid]

        self.text_goal_encoder = T5EncoderModel.from_pretrained("google/flan-t5-small")
        self.text_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.text_adapter = nn.Sequential(
            nn.Linear(512, self.goal_embed_dims), nn.LayerNorm(self.goal_embed_dims), nn.ReLU()
        )

        self.blind = self.dino_uuid not in observation_spaces.spaces
        if not self.blind:
            self.dino_tensor_shape = observation_spaces.spaces[self.dino_uuid].shape
            self.dino_compressor = nn.Sequential(
                nn.Conv2d(self.dino_tensor_shape[-1], self.dino_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.dino_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.dino_hid_out_dims[1] + self.goal_embed_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_embed_dims
        else:
            return (
                self.combine_hid_out_dims[-1]
                * self.dino_tensor_shape[0]
                * self.dino_tensor_shape[1]
            )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_dino(self, observations):
        return self.dino_compressor(observations[self.dino_uuid])

    def distribute_target(self, observations):
        max_len = observations[self.goal_uuid].shape[-1]
        goals_tensor = observations[self.goal_uuid].cpu().numpy().astype(np.uint8)
        goals = []
        for g in goals_tensor:
            g = convert_byte_to_string(g, max_len=max_len)
            goals.append(g)
        with torch.no_grad():
            goal_emb = self.text_tokenizer(goals, return_tensors="pt", padding=True).to(
                observations[self.goal_uuid].device
            )
            goal_emb = self.text_goal_encoder(**goal_emb).last_hidden_state
        goal_emb = self.text_adapter(goal_emb)
        goal_emb = torch.mean(goal_emb, dim=-2)
        return goal_emb.view(-1, self.goal_embed_dims, 1, 1).expand(
            -1, -1, self.dino_tensor_shape[-3], self.dino_tensor_shape[-2]
        )

    def adapt_input(self, observations):
        observations = {**observations}
        dino = observations[self.dino_uuid]
        goal = observations[self.goal_uuid]

        use_agent = False
        nagent = 1

        if len(dino.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = dino.shape[:3]
        else:
            nstep, nsampler = dino.shape[:2]

        observations[self.dino_uuid] = dino.view(-1, *dino.shape[-3:])
        observations[self.goal_uuid] = goal.view(-1, goal.shape[-1])

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(observations)

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        embs = [
            self.compress_dino(observations),
            self.distribute_target(observations),
        ]
        x = self.target_obs_combiner(
            torch.cat(
                embs,
                dim=1,
            )
        )
        x = x.reshape(x.size(0), -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)
