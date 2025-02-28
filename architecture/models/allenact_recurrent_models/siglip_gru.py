from collections import OrderedDict
from typing import Optional, List, Dict, cast, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
from allenact.algorithms.onpolicy_sync.policy import ObservationType, DistributionType
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.aux_losses.losses import MultiAuxTaskNegEntropyLoss
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact.embodiedai.models.visual_nav_models import (
    VisualNavActorCritic,
    FusionType,
)
from allenact.utils.model_utils import FeatureEmbedding
from allenact.utils.system import get_logger
from gym.spaces import Dict as SpaceDict
from open_clip import create_model_from_pretrained
from open_clip import get_tokenizer

from utils.string_utils import convert_byte_to_string
from architecture.models.allenact_transformer_models.allenact_dino_transformer import (
    DinoTxGoalEncoder,
)


class SigLIPTensorNavActorCritic(VisualNavActorCritic):
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
        rgb_siglip_preprocessor_uuid: Optional[str] = None,
        manipulation_rgb_siglip_preprocessor_uuid: Optional[str] = None,
        arm_proprioception_sensor_uuid: Optional[str] = None,
        an_object_is_in_hand_uuid: Optional[str] = None,
        goal_dims: int = 32,
        siglip_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        use_transformer_goal_encoder: bool = False,
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
        self.arm_proprioception_sensor_uuid = arm_proprioception_sensor_uuid
        self.an_object_is_in_hand_uuid = an_object_is_in_hand_uuid
        if self.an_object_is_in_hand_uuid is not None:
            self.hand_embedder = FeatureEmbedding(
                input_size=2,
                output_size=action_embed_size if an_object_is_in_hand_uuid is not None else 0,
            )

        self.use_transformer_goal_encoder = use_transformer_goal_encoder
        if use_transformer_goal_encoder:
            assert manipulation_rgb_siglip_preprocessor_uuid is None
            self.goal_visual_encoder = DinoTxGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                rgb_siglip_preprocessor_uuid,
                None,
                None,
                goal_dims,
                siglip_compressor_hidden_out_dims,
                combiner_hidden_out_dims=512,
                combiner_heads=8,
                combiner_layers=3,
            )
        else:
            self.goal_visual_encoder = SigLIPTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                rgb_siglip_preprocessor_uuid,
                manipulation_rgb_siglip_preprocessor_uuid,
                goal_dims,
                siglip_compressor_hidden_out_dims,
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

        if self.arm_proprioception_sensor_uuid is not None:
            self.proprioception_embedding = nn.Sequential(
                nn.Linear(4, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 32),
                nn.LeakyReLU(),
                nn.Linear(32, action_embed_size * 4),
            )

        self.train()

    def create_state_encoders(
        self,
        obs_embed_size: int,
        prev_action_embed_size: int,
        num_rnn_layers: int,
        rnn_type: str,
        add_prev_actions: bool,
        add_prev_action_null_token: bool,
        trainable_masked_hidden_state=False,
    ):
        rnn_input_size = obs_embed_size
        self.prev_action_embedder = FeatureEmbedding(
            input_size=int(add_prev_action_null_token) + self.action_space.n,
            output_size=prev_action_embed_size if add_prev_actions else 0,
        )
        if add_prev_actions:
            rnn_input_size += prev_action_embed_size
        if self.an_object_is_in_hand_uuid is not None:
            rnn_input_size += prev_action_embed_size
        if self.arm_proprioception_sensor_uuid is not None:
            rnn_input_size += prev_action_embed_size * 4

        state_encoders = OrderedDict()  # perserve insertion order in py3.6
        if self.multiple_beliefs:  # multiple belief model
            for aux_uuid in self.auxiliary_uuids:
                state_encoders[aux_uuid] = RNNStateEncoder(
                    rnn_input_size,
                    self._hidden_size,
                    num_layers=num_rnn_layers,
                    rnn_type=rnn_type,
                    trainable_masked_hidden_state=trainable_masked_hidden_state,
                )
            # create fusion model
            self.fusion_model = self.beliefs_fusion(
                hidden_size=self._hidden_size,
                obs_embed_size=obs_embed_size,
                num_tasks=len(self.auxiliary_uuids),
            )

        else:  # single belief model
            state_encoders["single_belief"] = RNNStateEncoder(
                rnn_input_size,
                self._hidden_size,
                num_layers=num_rnn_layers,
                rnn_type=rnn_type,
                trainable_masked_hidden_state=trainable_masked_hidden_state,
            )

        self.state_encoders = nn.ModuleDict(state_encoders)

        self.belief_names = list(self.state_encoders.keys())

        get_logger().info(
            "there are {} belief models: {}".format(len(self.belief_names), self.belief_names)
        )

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.goal_visual_encoder(observations)

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """

        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        if self.use_transformer_goal_encoder:
            obs_embeds, _ = self.forward_encoder(observations)
        else:
            obs_embeds = self.forward_encoder(observations)

        # 1.2 use embedding model to get prev_action embeddings
        if self.prev_action_embedder.input_size == self.action_space.n + 1:
            # In this case we have a unique embedding for the start of an episode
            prev_actions_embeds = self.prev_action_embedder(
                torch.where(
                    condition=0 != masks.view(*prev_actions.shape),
                    input=prev_actions + 1,
                    other=torch.zeros_like(prev_actions),
                )
            )
        else:
            prev_actions_embeds = self.prev_action_embedder(prev_actions)

        features_to_cat = [obs_embeds, prev_actions_embeds]
        if self.an_object_is_in_hand_uuid is not None:
            hand_embeds = self.hand_embedder(
                observations[self.an_object_is_in_hand_uuid].squeeze(2)
            )
            features_to_cat.append(hand_embeds)
        if self.arm_proprioception_sensor_uuid is not None:
            proprioception_embeds = self.proprioception_embedding(
                observations[self.arm_proprioception_sensor_uuid].float()
            )
            features_to_cat.append(proprioception_embeds)
        joint_embeds = torch.cat(features_to_cat, dim=-1)  # (T, N, *)

        # 2. use RNNs to get single/multiple beliefs
        beliefs_dict = {}
        for key, model in self.state_encoders.items():
            beliefs_dict[key], rnn_hidden_states = model(joint_embeds, memory.tensor(key), masks)
            memory.set_tensor(key, rnn_hidden_states)  # update memory here

        # 3. fuse beliefs for multiple belief models
        beliefs, task_weights = self.fuse_beliefs(beliefs_dict, obs_embeds)  # fused beliefs

        # 4. prepare output
        extras = (
            {
                aux_uuid: {
                    "beliefs": (beliefs_dict[aux_uuid] if self.multiple_beliefs else beliefs),
                    "obs_embeds": obs_embeds,
                    "aux_model": (
                        self.aux_models[aux_uuid] if aux_uuid in self.aux_models else None
                    ),
                }
                for aux_uuid in self.auxiliary_uuids
            }
            if self.auxiliary_uuids is not None
            else {}
        )

        if self.multiple_beliefs:
            extras[MultiAuxTaskNegEntropyLoss.UUID] = task_weights

        actor_critic_output = ActorCriticOutput(
            distributions=self.actor(beliefs),
            values=self.critic(beliefs),
            extras=extras,
        )

        return actor_critic_output, memory


class SigLIPTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        rgb_siglip_preprocessor_uuid: str,
        manipulation_siglip_preprocessor_uuid: str,
        goal_embed_dims: int = 32,
        siglip_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.rgb_siglip_uuid = rgb_siglip_preprocessor_uuid
        self.manipulation_siglip_uuid = manipulation_siglip_preprocessor_uuid
        self.goal_embed_dims = goal_embed_dims
        self.siglip_hid_out_dims = siglip_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.goal_space = observation_spaces.spaces[self.goal_uuid]

        self.text_goal_encoder = create_model_from_pretrained("hf-hub:timm/ViT-B-16-SigLIP-256")[
            0
        ].text
        self.text_goal_encoder.output_tokens = True
        self.text_tokenizer = get_tokenizer("hf-hub:timm/ViT-B-16-SigLIP-256")
        self.text_adapter = nn.Sequential(
            nn.Linear(768, self.goal_embed_dims), nn.LayerNorm(self.goal_embed_dims), nn.ReLU()
        )

        self.blind = False

        # navigation camera
        self.rgb_siglip_tensor_shape = observation_spaces.spaces[self.rgb_siglip_uuid].shape
        self.siglip_compressor = nn.Sequential(
            nn.Conv2d(self.rgb_siglip_tensor_shape[-1], self.siglip_hid_out_dims[0], 1),
            nn.ReLU(),
            nn.Conv2d(*self.siglip_hid_out_dims[0:2], 1),
            nn.ReLU(),
        )
        self.target_obs_combiner = nn.Sequential(
            nn.Conv2d(
                self.siglip_hid_out_dims[1] + self.goal_embed_dims,
                self.combine_hid_out_dims[0],
                1,
            ),
            nn.ReLU(),
            nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
        )

        if self.manipulation_siglip_uuid is not None:
            # manipulation camera
            self.manipulation_siglip_tensor_shape = observation_spaces.spaces[
                self.manipulation_siglip_uuid
            ].shape
            self.manipulation_siglip_compressor = nn.Sequential(
                nn.Conv2d(
                    self.manipulation_siglip_tensor_shape[-1], self.siglip_hid_out_dims[0], 1
                ),
                nn.ReLU(),
                nn.Conv2d(*self.siglip_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.manipulation_target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.siglip_hid_out_dims[1] + self.goal_embed_dims,
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
            if self.manipulation_siglip_uuid is not None:
                return (
                    self.combine_hid_out_dims[-1]
                    * self.rgb_siglip_tensor_shape[0]
                    * self.rgb_siglip_tensor_shape[1]
                    + self.combine_hid_out_dims[-1]
                    * self.manipulation_siglip_tensor_shape[0]
                    * self.manipulation_siglip_tensor_shape[1]
                )
            else:
                return (
                    self.combine_hid_out_dims[-1]
                    * self.rgb_siglip_tensor_shape[0]
                    * self.rgb_siglip_tensor_shape[1]
                )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_siglip(self, observations):
        return self.siglip_compressor(observations[self.rgb_siglip_uuid])

    def compress_manipulation_siglip(self, observations):
        return self.manipulation_siglip_compressor(observations[self.manipulation_siglip_uuid])

    def distribute_target(self, observations):
        max_len = observations[self.goal_uuid].shape[-1]
        goals_tensor = observations[self.goal_uuid].cpu().numpy().astype(np.uint8)
        goals = []
        for g in goals_tensor:
            g = convert_byte_to_string(g, max_len=max_len)
            goals.append(g)
        with torch.no_grad():
            goal_emb = self.text_tokenizer(goals, context_length=64).to(
                observations[self.goal_uuid].device
            )
            cls_feats, text_feats = self.text_goal_encoder(goal_emb)
            goal_emb = cls_feats
        goal_emb = self.text_adapter(goal_emb)
        return goal_emb.view(-1, self.goal_embed_dims, 1, 1).expand(
            -1, -1, self.rgb_siglip_tensor_shape[-3], self.rgb_siglip_tensor_shape[-2]
        )

    def adapt_input(self, observations):
        observations = {**observations}
        rgb_siglip = observations[self.rgb_siglip_uuid]
        if self.manipulation_siglip_uuid is not None:
            manipulation_rgb_siglip = observations[self.manipulation_siglip_uuid]
        goal = observations[self.goal_uuid]

        use_agent = False
        nagent = 1

        if len(rgb_siglip.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = rgb_siglip.shape[:3]
        else:
            nstep, nsampler = rgb_siglip.shape[:2]

        observations[self.rgb_siglip_uuid] = rgb_siglip.view(-1, *rgb_siglip.shape[-3:])
        if self.manipulation_siglip_uuid is not None:
            observations[self.manipulation_siglip_uuid] = manipulation_rgb_siglip.view(
                -1, *manipulation_rgb_siglip.shape[-3:]
            )
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

        text_feats = self.distribute_target(observations)
        embs = [
            self.compress_siglip(observations),
            text_feats,
        ]
        x = self.target_obs_combiner(
            torch.cat(
                embs,
                dim=1,
            )
        )
        x = x.reshape(x.size(0), -1)  # flatten
        if self.manipulation_siglip_uuid is not None:
            manipulation_embs = [
                self.compress_manipulation_siglip(observations),
                text_feats,
            ]
            y = self.manipulation_target_obs_combiner(
                torch.cat(
                    manipulation_embs,
                    dim=1,
                )
            )
            y = y.reshape(y.size(0), -1)  # flatten

            o = torch.cat([x, y], -1)
        else:
            o = x

        return self.adapt_output(o, use_agent, nstep, nsampler, nagent)
