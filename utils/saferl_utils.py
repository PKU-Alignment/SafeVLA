"""
Safe RL specific utilities and abstractions.

This module defines types and structures specific to safe reinforcement learning,
following the same design patterns as AllenAct's misc.py but for safety-aware RL.

This includes:
- SafeRLStepResult: Extended step result with cost signal
- SafePPOValue: Cost value loss for SafePPO algorithm
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    NamedTuple,
    Callable,
    cast,
    Generic,
    Sequence,
    Tuple,
)
import abc
import attr
import gym
import torch
import numpy as np
import json
import numbers
import os
import signal
import sys
import traceback
from multiprocessing.process import BaseProcess
import torch.distributed as dist
from multiprocessing.context import BaseContext
import torch.optim as optim
from setproctitle import setproctitle as ptitle
from omnisafe.common.lagrange import Lagrange
import torch.multiprocessing as mp  # type: ignore

try:
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from torch.optim.lr_scheduler import _LRScheduler
except (ImportError, ModuleNotFoundError):
    raise ImportError("`_LRScheduler` was not found in `torch.optim.lr_scheduler`")
from allenact.base_abstractions.misc import (
    RLStepResult,
    DistributionType,
    ActorCriticOutput,
    GenericAbstractLoss,
)
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
import allenact.utils.spaces_utils as su
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.utils.system import get_logger
from allenact.base_abstractions.misc import Memory
from allenact.algorithms.onpolicy_sync.policy import (
    FullMemorySpecType,
    ObservationType,
)
from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel

from allenact.utils import spaces_utils as su
from allenact.utils.tensor_utils import batch_observations, detach_recursively
from allenact.algorithms.onpolicy_sync.runner import OnPolicyRunner, OnPolicyTrainer, SaveDirFormat
from allenact.algorithms.onpolicy_sync.engine import (
    TRAIN_MODE_STR,
    VALID_MODE_STR,
    OnPolicyRLEngine,
)
from allenact.utils.experiment_utils import (
    PipelineStage,
    ScalarMeanTracker,
    StageComponent,
    TrainingPipeline,
    set_seed,
)
from allenact.algorithms.onpolicy_sync.storage import (
    StreamingStorageMixin,
    MiniBatchStorageMixin,
    RolloutBlockStorage as _BaseRolloutBlockStorage,
    ExperienceStorage,
)
from allenact.utils.experiment_utils import (
    set_seed,
    download_checkpoint_from_wandb,
    PipelineStage,
    StageComponent,
)
from allenact.utils.misc_utils import (
    NumpyJSONEncoder,
)
from allenact.utils.model_utils import md5_hash_of_state_dict
from allenact.utils.system import get_logger


class SafeRLStepResult(NamedTuple):
    """Extended RL step result for Safe RL with cost signal.

    This extends the standard RLStepResult to include a cost signal,
    which is used in constrained reinforcement learning algorithms
    like PPO-Lagrangian (SafePPO).

    # Attributes

    observation: The observation from the environment
    reward: The reward signal (for task success/completion)
    cost: The cost signal (for safety violations, e.g., collisions, dangerous interactions)
    done: Whether the episode is done
    info: Additional information dictionary

    # Example Usage

    ```python
    # In task.step()
    step_result = SafeRLStepResult(
        observation=self.get_observations(),
        reward=self.judge(),
        cost=self.compute_cost(),  # e.g., 1.0 if collision else 0.0
        done=self.is_done(),
        info={"last_action_success": True}
    )
    ```
    """

    observation: Optional[Any]
    reward: Optional[Union[float, List[float]]]
    cost: Optional[Union[float, List[float]]]
    done: Optional[bool]
    info: Optional[Dict[str, Any]]

    def clone(self, new_info: Dict[str, Any]) -> "SafeRLStepResult":
        """Create a new SafeRLStepResult with updated fields.

        # Parameters

        new_info: Dictionary with keys matching field names and their new values

        # Returns

        A new SafeRLStepResult with updated fields
        """
        return SafeRLStepResult(
            observation=(
                self.observation if "observation" not in new_info else new_info["observation"]
            ),
            reward=self.reward if "reward" not in new_info else new_info["reward"],
            cost=self.cost if "cost" not in new_info else new_info["cost"],
            done=self.done if "done" not in new_info else new_info["done"],
            info=self.info if "info" not in new_info else new_info["info"],
        )

    def merge(self, other: "SafeRLStepResult") -> "SafeRLStepResult":
        """Merge with another SafeRLStepResult, prioritizing non-None values from other.

        # Parameters

        other: Another SafeRLStepResult to merge with

        # Returns

        A new SafeRLStepResult with merged fields
        """
        return SafeRLStepResult(
            observation=(self.observation if other.observation is None else other.observation),
            reward=self.reward if other.reward is None else other.reward,
            cost=self.cost if other.cost is None else other.cost,
            done=self.done if other.done is None else other.done,
            info={
                **(self.info if self.info is not None else {}),
                **(other.info if other.info is not None else {}),
            },
        )

    def to_standard(self) -> RLStepResult:
        """Convert SafeRLStepResult to standard RLStepResult.

        The cost information is moved to the info dict under the key 'cost'.
        This is useful when interfacing with standard RL code that doesn't
        expect a cost field.

        # Returns

        A standard RLStepResult with cost stored in info dict
        """
        info = self.info.copy() if self.info is not None else {}
        info["cost"] = self.cost
        return RLStepResult(
            observation=self.observation,
            reward=self.reward,
            done=self.done,
            info=info,
        )


class SafePPOValue(AbstractActorCriticLoss):
    """Cost value loss for SafePPO (PPO-Lagrangian).

    This computes the value loss for the cost critic in safe reinforcement learning.
    It follows the same clipping mechanism as standard PPO value loss but applies
    to the cost signal instead of reward.

    # Attributes

    clip_param: The clipping parameter to use
    use_clipped_value_loss: Whether or not to also clip the value loss
    clip_decay: Optional function to decay clip_param over training steps

    # Example Usage

    ```python
    # In training config
    loss_config = SafePPOValue(
        clip_param=0.1,
        use_clipped_value_loss=True,
    )
    ```
    """

    def __init__(
        self,
        clip_param: float,
        use_clipped_value_loss: bool = True,
        clip_decay: Optional[Callable[[int], float]] = None,
        *args,
        **kwargs,
    ):
        """Initializer.

        # Parameters

        clip_param: The clipping parameter (typically 0.1 or 0.2)
        use_clipped_value_loss: Whether to clip the cost value loss
        clip_decay: Optional decay function for clip_param(step_count) -> scale
        """
        super().__init__(*args, **kwargs)
        self.clip_param = clip_param
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_decay = clip_decay if clip_decay is not None else (lambda x: 1.0)

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        """Compute the cost value loss.

        # Parameters

        step_count: Current training step count
        batch: Batch of data containing 'c_values' and 'c_returns'
        actor_critic_output: Output from actor-critic model containing c_values

        # Returns

        Tuple of (loss_value, info_dict)
        """
        c_values = actor_critic_output.c_values
        clip_param = self.clip_param * self.clip_decay(step_count)

        if self.use_clipped_value_loss:
            # Clipped cost value loss (PPO-style)
            c_value_pred_clipped = batch["c_values"] + (c_values - batch["c_values"]).clamp(
                -clip_param, clip_param
            )
            c_value_losses = (c_values - batch["c_returns"]).pow(2)
            c_value_losses_clipped = (c_value_pred_clipped - batch["c_returns"]).pow(2)
            c_value_loss = 0.5 * torch.max(c_value_losses, c_value_losses_clipped).mean()
        else:
            # Unclipped cost value loss (standard MSE)
            c_value_loss = (
                0.5 * (cast(torch.FloatTensor, batch["c_returns"]) - c_values).pow(2).mean()
            )

        return (
            c_value_loss,
            {
                "c_value": c_value_loss.item(),
            },
        )


# Default SafePPO configuration
SafePPOConfig = dict(
    clip_param=0.1,
    value_loss_coef=0.5,
    entropy_coef=0.01,
)


class SafeActorCriticOutput(tuple, Generic[DistributionType]):
    distributions: DistributionType
    values: torch.FloatTensor
    c_values: torch.FloatTensor
    extras: Dict[str, Any]

    # noinspection PyTypeChecker
    def __new__(
        cls,
        distributions: DistributionType,
        values: torch.FloatTensor,
        c_values: torch.FloatTensor,
        extras: Dict[str, Any],
    ):
        self = tuple.__new__(cls, (distributions, values, c_values, extras))
        self.distributions = distributions
        self.values = values
        self.c_values = c_values
        self.extras = extras
        return self

    def __repr__(self) -> str:
        return (
            f"Group(distributions={self.distributions},"
            f" values={self.values},"
            f" c_values={self.c_values},"
            f" extras={self.extras})"
        )


class SafeExperienceStorage(abc.ABC):
    @abc.abstractmethod
    def initialize(self, *, observations: ObservationType, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def add(
        self,
        observations: ObservationType,
        memory: Optional[Memory],
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        c_value_preds: torch.Tensor,
        rewards: torch.Tensor,
        costs: torch.Tensor,
        masks: torch.Tensor,
    ):
        """
        # Parameters
        observations : Observations after taking `actions`
        memory: Memory after having observed the last set of observations.
        actions: Actions taken to reach the current state, i.e. taking these actions has led to a new state with
            new `observations`.
        action_log_probs : Log probs of `actions`
        value_preds : Value predictions corresponding to the last observations
            (i.e. the states before taking `actions`).
        rewards : Rewards from taking `actions` in the last set of states.
        masks : Masks corresponding to the current states, having 0 entries where `observations` correspond to
            observations from the beginning of a new episode.
        """
        raise NotImplementedError

    def before_updates(self, **kwargs):
        pass

    def after_updates(self, **kwargs) -> int:
        pass

    @abc.abstractmethod
    def to(self, device: torch.device):
        pass

    @abc.abstractmethod
    def set_partition(self, index: int, num_parts: int):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def total_experiences(self) -> int:
        raise NotImplementedError


class SafeRolloutStorage(SafeExperienceStorage, abc.ABC):
    # noinspection PyMethodOverriding
    @abc.abstractmethod
    def initialize(
        self,
        *,
        observations: ObservationType,
        num_samplers: int,
        recurrent_memory_specification: FullMemorySpecType,
        action_space: gym.Space,
        **kwargs,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def agent_input_for_next_step(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def sampler_select(self, keep_list: Sequence[int]):
        raise NotImplementedError


class SafeRolloutBlockStorage(_BaseRolloutBlockStorage, SafeRolloutStorage):
    """Extended RolloutBlockStorage that supports cost values for SafeRL."""

    def __init__(self, init_size: int = 50):
        super().__init__(init_size=init_size)
        self._c_value_preds_full: Optional[torch.Tensor] = None
        self._c_returns_full: Optional[torch.Tensor] = None
        self._costs_full: Optional[torch.Tensor] = None
        self._c_advantages: Optional[torch.Tensor] = None
        self._c_normalized_advantages: Optional[torch.Tensor] = None

    @property
    def c_value_preds(self) -> torch.Tensor:
        return self._c_value_preds_full[: self.step + 1]

    @property
    def costs(self) -> torch.Tensor:
        return self._costs_full[: self.step]

    @property
    def c_returns(self) -> torch.Tensor:
        return self._c_returns_full[: self.step + 1]

    def add(
        self,
        observations: ObservationType,
        memory: Optional[Memory],
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        c_value_preds: torch.Tensor,
        rewards: torch.Tensor,
        costs: torch.Tensor,
        masks: torch.Tensor,
    ):
        """Extended add method that includes cost value predictions and costs."""
        assert (
            len(masks.shape) == 2 and masks.shape[1] == 1
        ), f"Can only add a single step worth of data at a time (mask shape = {masks.shape})."

        self.total_experiences += masks.shape[0]

        if self.step == self.full_size:
            self._double_storage_size()
        elif self.step > self.full_size:
            raise RuntimeError

        self.insert_observations(observations, time_step=self.step + 1)
        self.insert_memory(memory, time_step=self.step + 1)

        assert actions.shape == self._actions_full.shape[1:]

        self._actions_full[self.step].copy_(actions)
        self._prev_actions_full[self.step + 1].copy_(actions)
        self._masks_full[self.step + 1].copy_(masks)

        if self._rewards_full is None:
            # Initialize storage for rewards, costs, value predictions, etc.
            self._rewards_full = self.create_tensor_storage(self.full_size, rewards.unsqueeze(0))

            self._costs_full = self.create_tensor_storage(self.full_size, costs.unsqueeze(0))

            value_returns_template = value_preds.unsqueeze(0)
            self._value_preds_full = self.create_tensor_storage(
                self.full_size + 1, value_returns_template
            )

            c_value_returns_template = c_value_preds.unsqueeze(0)
            self._c_value_preds_full = self.create_tensor_storage(
                self.full_size + 1, c_value_returns_template
            )

            self._returns_full = self.create_tensor_storage(
                self.full_size + 1, value_returns_template
            )

            self._c_returns_full = self.create_tensor_storage(
                self.full_size + 1, c_value_returns_template
            )

            self._action_log_probs_full = self.create_tensor_storage(
                self.full_size, action_log_probs.unsqueeze(0)
            )

        self._rewards_full[self.step].copy_(rewards)
        self._costs_full[self.step].copy_(costs)
        self._value_preds_full[self.step].copy_(value_preds)
        self._c_value_preds_full[self.step].copy_(c_value_preds)
        self._action_log_probs_full[self.step].copy_(action_log_probs)

        self.step = (self.step + 1) % self.full_size

    def _double_storage_size(self):
        """Double the storage size when capacity is reached."""

        def pad_tensor_with_zeros(old_t: torch.Tensor) -> torch.Tensor:
            padded_t = torch.zeros(
                (old_t.shape[0] * 2, *old_t.shape[1:]),
                dtype=old_t.dtype,
                device=old_t.device,
            )
            padded_t[: old_t.shape[0]] = old_t
            return padded_t

        # Call parent's double storage size for base attributes
        super()._double_storage_size()

        # Double storage for cost-related attributes
        if self._costs_full is not None:
            self._costs_full = pad_tensor_with_zeros(self._costs_full)
        if self._c_value_preds_full is not None:
            self._c_value_preds_full = pad_tensor_with_zeros(self._c_value_preds_full)
        if self._c_returns_full is not None:
            self._c_returns_full = pad_tensor_with_zeros(self._c_returns_full)

    def compute_returns(
        self,
        next_value: torch.Tensor,
        next_c_value: torch.Tensor,
        use_gae: bool,
        gamma: float,
        tau: float,
    ):
        """Compute returns and cost returns using GAE or simple discounting."""

        def _extend_tensor_with_ones(stored_tensor: torch.Tensor, desired_num_dims: int):
            extended_shape = stored_tensor.shape + (1,) * (
                desired_num_dims - len(stored_tensor.shape)
            )
            return stored_tensor.view(*extended_shape)

        extended_mask = _extend_tensor_with_ones(
            self.masks, desired_num_dims=len(self.value_preds.shape)
        )
        extended_rewards = _extend_tensor_with_ones(
            self.rewards, desired_num_dims=len(self.value_preds.shape)
        )
        extended_costs = _extend_tensor_with_ones(
            self.costs, desired_num_dims=len(self.c_value_preds.shape)
        )

        if use_gae:
            self.value_preds[-1] = next_value
            self.c_value_preds[-1] = next_c_value
            gae = 0
            c_gae = 0
            for step in reversed(range(extended_rewards.shape[0])):
                delta = (
                    extended_rewards[step]
                    + gamma * self.value_preds[step + 1] * extended_mask[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * extended_mask[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]

                c_delta = (
                    extended_costs[step]
                    + gamma * self.c_value_preds[step + 1] * extended_mask[step + 1]
                    - self.c_value_preds[step]
                )
                c_gae = c_delta + gamma * tau * extended_mask[step + 1] * c_gae
                self.c_returns[step] = c_gae + self.c_value_preds[step]
        else:
            self.returns[-1] = next_value
            self.c_returns[-1] = next_c_value
            for step in reversed(range(extended_rewards.shape[0])):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * extended_mask[step + 1]
                    + extended_rewards[step]
                )
                self.c_returns[step] = (
                    self.c_returns[step + 1] * gamma * extended_mask[step + 1]
                    + extended_costs[step]
                )

    def before_updates(
        self,
        *,
        next_value: torch.Tensor,
        next_c_value: Optional[torch.Tensor] = None,
        use_gae: bool,
        gamma: float,
        tau: float,
        adv_stats_callback: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
        **kwargs,
    ):
        """Compute advantages and normalized advantages before updates."""
        assert len(kwargs) == 0

        # If next_c_value is not provided, use zeros (for compatibility with non-SafeRL code)
        if next_c_value is None:
            next_c_value = torch.zeros_like(next_value)

        self.compute_returns(
            next_value=next_value,
            next_c_value=next_c_value,
            use_gae=use_gae,
            gamma=gamma,
            tau=tau,
        )

        self._advantages = self.returns[:-1] - self.value_preds[:-1]
        self._c_advantages = self.c_returns[:-1] - self.c_value_preds[:-1]

        adv_stats = adv_stats_callback(self._advantages)
        self._normalized_advantages = (self._advantages - adv_stats["mean"]) / (
            adv_stats["std"] + 1e-5
        )

        c_adv_stats = adv_stats_callback(self._c_advantages)
        self._c_normalized_advantages = (self._c_advantages - c_adv_stats["mean"]) / (
            c_adv_stats["std"] + 1e-5
        )

        self._before_update_called = True

    def after_updates(self, **kwargs):
        """Clean up after updates and reset for next rollout."""
        assert len(kwargs) == 0

        for storage in [self.observations, self.memory_first_last]:
            for key in storage:
                storage[key][0][0].copy_(storage[key][0][-1])

        if self._masks_full is not None:
            self.masks[0].copy_(self.masks[-1])

        if self._prev_actions_full is not None:
            self.prev_actions[0].copy_(self.prev_actions[-1])

        self._before_update_called = False
        self._advantages = None
        self._normalized_advantages = None
        self._c_advantages = None
        self._c_normalized_advantages = None
        self.step = 0

    def batched_experience_generator(self, num_mini_batch: int):
        """Generate batched experience including cost-related data."""
        import random
        import numpy as np

        assert self._before_update_called, (
            "self._before_update_called() must be called before"
            " attempting to generated batched rollouts."
        )
        num_samplers = self.rewards.shape[1]
        assert num_samplers >= num_mini_batch, (
            f"The number of task samplers ({num_samplers}) "
            f"must be greater than or equal to the number of "
            f"mini batches ({num_mini_batch})."
        )

        inds = np.round(np.linspace(0, num_samplers, num_mini_batch + 1, endpoint=True)).astype(
            np.int32
        )
        pairs = list(zip(inds[:-1], inds[1:]))
        random.shuffle(pairs)

        for start_ind, end_ind in pairs:
            cur_samplers = list(range(start_ind, end_ind))

            memory_batch = self.memory_first_last.step_squeeze(0).sampler_select(cur_samplers)
            observations_batch = self.unflatten_observations(
                self.observations.slice(dim=0, stop=-1).sampler_select(cur_samplers)
            )

            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            c_value_preds_batch = []
            return_batch = []
            c_return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            norm_adv_targ = []
            c_adv_targ = []
            c_norm_adv_targ = []

            for ind in cur_samplers:
                actions_batch.append(self.actions[:, ind])
                prev_actions_batch.append(self.prev_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                c_value_preds_batch.append(self.c_value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                c_return_batch.append(self.c_returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])

                adv_targ.append(self._advantages[:, ind])
                c_adv_targ.append(self._c_advantages[:, ind])
                norm_adv_targ.append(self._normalized_advantages[:, ind])
                c_norm_adv_targ.append(self._c_normalized_advantages[:, ind])

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            c_value_preds_batch = torch.stack(c_value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            c_return_batch = torch.stack(c_return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)
            norm_adv_targ = torch.stack(norm_adv_targ, 1)
            c_adv_targ = torch.stack(c_adv_targ, 1)
            c_norm_adv_targ = torch.stack(c_norm_adv_targ, 1)

            yield {
                "observations": observations_batch,
                "memory": memory_batch,
                "actions": su.unflatten(self.action_space, actions_batch),
                "prev_actions": su.unflatten(self.action_space, prev_actions_batch),
                "values": value_preds_batch,
                "c_values": c_value_preds_batch,
                "returns": return_batch,
                "c_returns": c_return_batch,
                "masks": masks_batch,
                "old_action_log_probs": old_action_log_probs_batch,
                "adv_targ": adv_targ,
                "c_adv_targ": c_adv_targ,
                "norm_adv_targ": norm_adv_targ,
                "c_norm_adv_targ": c_norm_adv_targ,
                "bsize": int(np.prod(masks_batch.shape[:2])),
            }


@attr.s(kw_only=True)
class InferenceAgent:
    actor_critic: ActorCriticModel = attr.ib()
    rollout_storage: SafeRolloutStorage = attr.ib()
    device: torch.device = attr.ib()
    sensor_preprocessor_graph: Optional[SensorPreprocessorGraph] = attr.ib()
    steps_before_rollout_refresh: int = attr.ib(default=128)
    memory: Optional[Memory] = attr.ib(default=None)
    steps_taken_in_task: int = attr.ib(default=0)
    last_action_flat: Optional = attr.ib(default=None)
    has_initialized: Optional = attr.ib(default=False)

    def __attrs_post_init__(self):
        self.actor_critic.eval()
        self.actor_critic.to(device=self.device)
        if self.memory is not None:
            self.memory.to(device=self.device)
        if self.sensor_preprocessor_graph is not None:
            self.sensor_preprocessor_graph.to(self.device)

        self.rollout_storage.to(self.device)
        self.rollout_storage.set_partition(index=0, num_parts=1)

    @classmethod
    def from_experiment_config(
        cls,
        exp_config: ExperimentConfig,
        device: torch.device,
        checkpoint_path: Optional[str] = None,
        model_state_dict: Optional[Dict[str, Any]] = None,
        mode: str = "test",
    ):
        assert (
            checkpoint_path is None or model_state_dict is None
        ), "Cannot have `checkpoint_path` and `model_state_dict` both non-None."
        rollout_storage = exp_config.training_pipeline().rollout_storage

        machine_params = exp_config.machine_params(mode)
        if not isinstance(machine_params, MachineParams):
            machine_params = MachineParams(**machine_params)

        sensor_preprocessor_graph = machine_params.sensor_preprocessor_graph

        actor_critic = cast(
            ActorCriticModel,
            exp_config.create_model(sensor_preprocessor_graph=sensor_preprocessor_graph),
        )

        if checkpoint_path is not None:
            actor_critic.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
            )
        elif model_state_dict is not None:
            actor_critic.load_state_dict(
                model_state_dict
                if "model_state_dict" not in model_state_dict
                else model_state_dict["model_state_dict"]
            )

        return cls(
            actor_critic=actor_critic,
            rollout_storage=rollout_storage,
            device=device,
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    def reset(self):
        if self.has_initialized:
            self.rollout_storage.after_updates()
        self.steps_taken_in_task = 0
        self.memory = None

    def act(self, observations: ObservationType):
        # Batch of size 1
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
        self.last_action_flat = su.flatten(self.actor_critic.action_space, action)

        self.steps_taken_in_task += 1

        if self.steps_taken_in_task % self.steps_before_rollout_refresh == 0:
            self.rollout_storage.after_updates()

        return su.action_list(self.actor_critic.action_space, self.last_action_flat)[0]


class SafeOnPolicyRLEngine(OnPolicyRLEngine):

    def collect_step_across_all_task_samplers(
        self,
        rollout_storage_uuid: str,
        uuid_to_storage: Dict[str, SafeExperienceStorage],
        visualizer=None,
        dist_wrapper_class=None,
    ) -> int:
        """Collect a step across all task samplers with cost signal support.

        This method extends the base implementation to handle cost signals
        from SafeRLStepResult in addition to standard rewards.
        """
        rollout_storage = cast(SafeRolloutStorage, uuid_to_storage[rollout_storage_uuid])
        actions, actor_critic_output, memory, _ = self.act(
            rollout_storage=rollout_storage,
            dist_wrapper_class=dist_wrapper_class,
        )

        # Flatten actions
        flat_actions = su.flatten(self.actor_critic.action_space, actions)

        assert len(flat_actions.shape) == 3, (
            "Distribution samples must include step and task sampler dimensions [step, sampler, ...]. The simplest way"
            "to accomplish this is to pass param tensors (like `logits` in a `CategoricalDistr`) with these dimensions"
            "to the Distribution."
        )

        # Convert flattened actions into list of actions and send them
        from allenact.base_abstractions.misc import RLStepResult
        from allenact.algorithms.onpolicy_sync.vector_sampled_tasks import (
            COMPLETE_TASK_CALLBACK_KEY,
            COMPLETE_TASK_METRICS_KEY,
        )

        outputs: List[RLStepResult] = self.vector_tasks.step(
            su.action_list(self.actor_critic.action_space, flat_actions)
        )

        # Save after task completion metrics
        for step_result in outputs:
            if step_result.info is not None:
                if COMPLETE_TASK_METRICS_KEY in step_result.info:
                    new_metrics = step_result.info[COMPLETE_TASK_METRICS_KEY]
                    if hasattr(self, "_lagrange"):
                        new_metrics["lagrangian_multiplier"] = (
                            self._lagrange.lagrangian_multiplier.item()
                        )
                    self.single_process_metrics.append(new_metrics)
                    del step_result.info[COMPLETE_TASK_METRICS_KEY]
                if COMPLETE_TASK_CALLBACK_KEY in step_result.info:
                    self.single_process_task_callback_data.append(
                        step_result.info[COMPLETE_TASK_CALLBACK_KEY]
                    )
                    del step_result.info[COMPLETE_TASK_CALLBACK_KEY]

        rewards: Union[List, torch.Tensor]
        costs: Union[List, torch.Tensor]

        observations, rewards, costs, dones, infos = [list(x) for x in zip(*outputs)]

        rewards = torch.tensor(
            rewards,
            dtype=torch.float,
            device=self.device,
        )

        costs = torch.tensor(
            costs,
            dtype=torch.float,
            device=self.device,
        )

        # We want rewards to have dimensions [sampler, reward]
        if len(rewards.shape) == 1:
            # Rewards are of shape [sampler,]
            rewards = rewards.unsqueeze(-1)
        elif len(rewards.shape) > 1:
            raise NotImplementedError()

        if len(costs.shape) == 1:
            # Costs are of shape [sampler,]
            costs = costs.unsqueeze(-1)
        elif len(costs.shape) > 1:
            raise NotImplementedError()

        # If done then clean the history of observations.
        masks = (
            1.0
            - torch.tensor(
                dones,
                dtype=torch.float32,
                device=self.device,
            )
        ).view(-1, 1)

        npaused, keep, batch = self.remove_paused(observations)

        if hasattr(self.actor_critic, "sampler_select"):
            self.actor_critic.sampler_select(keep)

        if npaused > 0:
            if self.mode == TRAIN_MODE_STR:
                raise NotImplementedError(
                    "When trying to get a new task from a task sampler (using the `.next_task()` method)"
                    " the task sampler returned `None`. This is not currently supported during training"
                    " (and almost certainly a bug in the implementation of the task sampler or in the "
                    " initialization of the task sampler for training)."
                )

            for s in uuid_to_storage.values():
                if isinstance(s, SafeRolloutStorage):
                    s.sampler_select(keep)

        to_add_to_storage = dict(
            observations=(self._preprocess_observations(batch) if len(keep) > 0 else batch),
            memory=self._active_memory(memory, keep),
            actions=flat_actions[0, keep],
            action_log_probs=actor_critic_output.distributions.log_prob(actions)[0, keep],
            value_preds=actor_critic_output.values[0, keep],
            c_value_preds=actor_critic_output.c_values[0, keep],
            rewards=rewards[keep],
            costs=costs[keep],
            masks=masks[keep],
        )
        for storage in uuid_to_storage.values():
            storage.add(**to_add_to_storage)

        if visualizer is not None:
            if len(keep) > 0:
                visualizer.collect(
                    rollout=rollout_storage,
                    vector_task=self.vector_tasks,
                    alive=keep,
                    actor_critic=actor_critic_output,
                )
            else:
                visualizer.collect(actor_critic=actor_critic_output)

        return npaused

    def compute_losses_track_them_and_backprop(
        self,
        stage: PipelineStage,
        stage_component: StageComponent,
        storage: SafeExperienceStorage,
        skip_backprop: bool = False,
    ):
        """Compute losses with Lagrangian multiplier support for safe RL.

        This method extends the base implementation to:
        1. Update the Lagrangian multiplier based on cost constraints
        2. Pass cost-related parameters to loss functions
        """
        from allenact.algorithms.onpolicy_sync.misc import TrackingInfo, TrackingInfoType

        training = self.mode == TRAIN_MODE_STR

        assert training or skip_backprop

        if training and self.is_distributed:
            self.insufficient_data_for_update.set("insufficient_data_for_update", str(0))
            dist.barrier(
                device_ids=(None if self.device == torch.device("cpu") else [self.device.index])
            )

        # Update Lagrangian multiplier based on costs
        costs = self.training_pipeline.current_stage_storage[
            self.training_pipeline.rollout_storage_uuid
        ].costs
        costs_summed_over_steps = costs.sum(dim=0)
        costs_mean = costs_summed_over_steps.mean()

        if hasattr(self, "_lagrange"):
            self._lagrange.update_lagrange_multiplier(costs_mean)

        training_settings = stage_component.training_settings

        loss_names = stage_component.loss_names
        losses = [self.training_pipeline.get_loss(ln) for ln in loss_names]
        loss_weights = [stage.uuid_to_loss_weight[ln] for ln in loss_names]
        loss_update_repeats_list = training_settings.update_repeats
        if isinstance(loss_update_repeats_list, numbers.Integral):
            loss_update_repeats_list = [loss_update_repeats_list] * len(loss_names)

        if skip_backprop and isinstance(storage, MiniBatchStorageMixin):
            if loss_update_repeats_list != [1] * len(loss_names):
                loss_update_repeats_list = [1] * len(loss_names)
                get_logger().warning(
                    "Does not make sense to do multiple updates when"
                    " skip_backprop is `True` and you are using a storage of type"
                    " `MiniBatchStorageMixin`. This is likely a problem caused by"
                    " using a custom valid/test stage component that is inheriting its"
                    " TrainingSettings from the TrainingPipeline's TrainingSettings. We will override"
                    " the requested number of updates repeats (which was"
                    f" {dict(zip(loss_names, loss_update_repeats_list))}) to be 1 for all losses."
                )

        enough_data_for_update = True
        for current_update_repeat_index in range(max(loss_update_repeats_list, default=0)):
            if isinstance(storage, MiniBatchStorageMixin):
                batch_iterator = storage.batched_experience_generator(
                    num_mini_batch=training_settings.num_mini_batch
                )
            elif isinstance(storage, StreamingStorageMixin):
                assert (
                    training_settings.num_mini_batch is None
                    or training_settings.num_mini_batch == 1
                )

                def single_batch_generator(streaming_storage: StreamingStorageMixin):
                    try:
                        yield cast(StreamingStorageMixin, streaming_storage).next_batch()
                    except EOFError:
                        if not training:
                            raise

                        if streaming_storage.empty():
                            yield None
                        else:
                            cast(StreamingStorageMixin, streaming_storage).reset_stream()
                            stage.stage_component_uuid_to_stream_memory[
                                stage_component.uuid
                            ].clear()
                            yield cast(StreamingStorageMixin, streaming_storage).next_batch()

                batch_iterator = single_batch_generator(streaming_storage=storage)
            else:
                raise NotImplementedError(
                    f"Storage {storage} must be a subclass of `MiniBatchStorageMixin` or `StreamingStorageMixin`."
                )

            for batch in batch_iterator:
                if batch is None:
                    if training:
                        assert isinstance(storage, StreamingStorageMixin)
                        get_logger().warning(
                            f"Worker {self.worker_id}: could not run update in {storage}, potentially because"
                            f" not enough data has been accumulated to be able to fill an initial batch."
                        )
                    else:
                        pass
                    enough_data_for_update = False

                if training and self.is_distributed:
                    self.insufficient_data_for_update.add(
                        "insufficient_data_for_update",
                        1 * (not enough_data_for_update),
                    )
                    dist.barrier(
                        device_ids=(
                            None if self.device == torch.device("cpu") else [self.device.index]
                        )
                    )

                    if (
                        int(self.insufficient_data_for_update.get("insufficient_data_for_update"))
                        != 0
                    ):
                        enough_data_for_update = False
                        break

                info: Dict[str, float] = {}

                bsize: Optional[int] = None
                total_loss: Optional[torch.Tensor] = None
                actor_critic_output_for_batch: Optional[ActorCriticOutput] = None
                batch_memory = Memory()

                for loss, loss_name, loss_weight, max_update_repeats_for_loss in zip(
                    losses, loss_names, loss_weights, loss_update_repeats_list
                ):
                    if current_update_repeat_index >= max_update_repeats_for_loss:
                        continue

                    if isinstance(loss, AbstractActorCriticLoss):
                        bsize = batch["bsize"]

                        if actor_critic_output_for_batch is None:
                            try:
                                actor_critic_output_for_batch, _ = self.actor_critic(
                                    observations=batch["observations"],
                                    memory=batch["memory"],
                                    prev_actions=batch["prev_actions"],
                                    masks=batch["masks"],
                                )
                            except ValueError:
                                save_path = self.save_error_data(batch=batch)
                                get_logger().error(
                                    f"Encountered a value error! Likely because of nans in the output/input."
                                    f" Saving all error information to {save_path}."
                                )
                                raise

                        # Pass additional SafeRL parameters to loss function
                        loss_kwargs = dict(
                            step_count=self.step_count,
                            batch=batch,
                            actor_critic_output=actor_critic_output_for_batch,
                        )

                        # Add Lagrangian-related parameters if available
                        if hasattr(self, "_lagrange"):
                            loss_kwargs.update(
                                lagrangian_multiplier=self._lagrange.lagrangian_multiplier,
                                cost_limit=self._lagrange.cost_limit,
                                lambda_lr=self._lagrange.lambda_lr,
                                ep_costs=costs_mean,
                            )

                        loss_return = loss.loss(**loss_kwargs)

                        per_epoch_info = {}
                        if len(loss_return) == 2:
                            current_loss, current_info = loss_return
                        elif len(loss_return) == 3:
                            current_loss, current_info, per_epoch_info = loss_return
                        else:
                            raise NotImplementedError

                    elif isinstance(loss, GenericAbstractLoss):
                        loss_output = loss.loss(
                            model=self.actor_critic,
                            batch=batch,
                            batch_memory=batch_memory,
                            stream_memory=stage.stage_component_uuid_to_stream_memory[
                                stage_component.uuid
                            ],
                        )
                        current_loss = loss_output.value
                        current_info = loss_output.info
                        per_epoch_info = loss_output.per_epoch_info
                        batch_memory = loss_output.batch_memory
                        stage.stage_component_uuid_to_stream_memory[stage_component.uuid] = (
                            loss_output.stream_memory
                        )
                        bsize = loss_output.bsize
                    else:
                        raise NotImplementedError(
                            f"Loss of type {type(loss)} is not supported. Losses must be subclasses of"
                            f" `AbstractActorCriticLoss` or `GenericAbstractLoss`."
                        )

                    if total_loss is None:
                        total_loss = loss_weight * current_loss
                    else:
                        total_loss = total_loss + loss_weight * current_loss

                    for key, value in current_info.items():
                        info[f"{loss_name}/{key}"] = value

                    if per_epoch_info is not None:
                        for key, value in per_epoch_info.items():
                            if max(loss_update_repeats_list, default=0) > 1:
                                info[
                                    f"{loss_name}/{key}_epoch{current_update_repeat_index:02d}"
                                ] = value
                                info[f"{loss_name}/{key}_combined"] = value
                            else:
                                info[f"{loss_name}/{key}"] = value

                assert total_loss is not None, (
                    f"No {stage_component.uuid} losses specified for training in stage"
                    f" {self.training_pipeline.current_stage_index}"
                )

                total_loss_scalar = total_loss.item()
                info[f"total_loss"] = total_loss_scalar

                self.tracking_info_list.append(
                    TrackingInfo(
                        type=TrackingInfoType.LOSS,
                        info=info,
                        n=bsize,
                        storage_uuid=stage_component.storage_uuid,
                        stage_component_uuid=stage_component.uuid,
                    )
                )

                to_track = {
                    "rollout_epochs": max(loss_update_repeats_list, default=0),
                    "worker_batch_size": bsize,
                }

                aggregate_bsize = None
                if training:
                    aggregate_bsize = self.distributed_weighted_sum(bsize, 1)
                    to_track["global_batch_size"] = aggregate_bsize
                    to_track["lr"] = self.optimizer.param_groups[0]["lr"]

                if training_settings.num_mini_batch is not None:
                    to_track["rollout_num_mini_batch"] = training_settings.num_mini_batch

                for k, v in to_track.items():
                    self.tracking_info_list.append(
                        TrackingInfo(
                            type=TrackingInfoType.UPDATE_INFO,
                            info={k: v},
                            n=1 if k == "worker_batch_size" else bsize,
                            storage_uuid=stage_component.storage_uuid,
                            stage_component_uuid=stage_component.uuid,
                        )
                    )

                if not skip_backprop:
                    total_grad_norm = self.backprop_step(
                        total_loss=total_loss,
                        max_grad_norm=training_settings.max_grad_norm,
                        local_to_global_batch_size_ratio=bsize / aggregate_bsize,
                    )
                    self.tracking_info_list.append(
                        TrackingInfo(
                            type=TrackingInfoType.UPDATE_INFO,
                            info={"total_grad_norm": total_grad_norm},
                            n=bsize,
                            storage_uuid=stage_component.storage_uuid,
                            stage_component_uuid=stage_component.uuid,
                        )
                    )

                stage.stage_component_uuid_to_stream_memory[stage_component.uuid] = (
                    detach_recursively(
                        input=stage.stage_component_uuid_to_stream_memory[stage_component.uuid],
                        inplace=True,
                    )
                )

    def run_pipeline(self, valid_on_initial_weights: bool = False):
        cur_stage_training_settings = self.training_pipeline.current_stage.training_settings

        # Change engine attributes that depend on the current stage
        self.training_pipeline.current_stage.change_engine_attributes(self)

        rollout_storage = self.training_pipeline.rollout_storage
        uuid_to_storage = self.training_pipeline.current_stage_storage
        self.initialize_storage_and_viz(
            storage_to_initialize=cast(List[ExperienceStorage], list(uuid_to_storage.values()))
        )
        self.tracking_info_list.clear()

        self.last_log = self.training_pipeline.total_steps

        if self.last_save is None:
            self.last_save = self.training_pipeline.total_steps

        should_save_checkpoints = (
            self.checkpoints_dir != ""
            and cur_stage_training_settings.save_interval is not None
            and cur_stage_training_settings.save_interval > 0
        )
        already_saved_checkpoint = False

        if (
            valid_on_initial_weights
            and should_save_checkpoints
            and self.checkpoints_queue is not None
        ):
            if (
                self.save_ckpt_at_every_host and self.worker_id == self.first_local_worker_id
            ) or self.worker_id == 0:
                model_path = self.checkpoint_save()
                if self.checkpoints_queue is not None:
                    self.checkpoints_queue.put(("eval", model_path))

        while True:
            pipeline_stage_changed = self.training_pipeline.before_rollout(
                train_metrics=self._last_aggregated_train_task_metrics
            )  # This is `False` at the very start of training, i.e. pipeline starts with a stage initialized

            self._last_aggregated_train_task_metrics.reset()
            training_is_complete = self.training_pipeline.current_stage is None

            # `training_is_complete` should imply `pipeline_stage_changed`
            assert pipeline_stage_changed or not training_is_complete

            #  Saving checkpoints and initializing storage when the pipeline stage changes
            if pipeline_stage_changed:
                # Here we handle saving a checkpoint after a pipeline stage ends. We
                # do this:
                # (1) after every pipeline stage if the `self.save_ckpt_after_every_pipeline_stage`
                #   boolean is True, and
                # (2) when we have reached the end of ALL training (i.e. all stages are complete).
                if (
                    should_save_checkpoints
                    and (  # Might happen if the `save_interval` was hit just previously, see below
                        not already_saved_checkpoint
                    )
                    and (self.save_ckpt_after_every_pipeline_stage or training_is_complete)
                ):
                    self._save_checkpoint_then_send_checkpoint_for_validation_and_update_last_save_counter(
                        pipeline_stage_index=(
                            self.training_pipeline.current_stage_index - 1
                            if not training_is_complete
                            else len(self.training_pipeline.pipeline_stages) - 1
                        )
                    )

                # If training is complete, break out
                if training_is_complete:
                    break

                # Here we handle updating our training settings after a pipeline stage ends.
                # Update the training settings we're using
                cur_stage_training_settings = self.training_pipeline.current_stage.training_settings

                # If the pipeline stage changed we must initialize any new custom storage and
                # stop updating any custom storage that is no longer in use (this second bit
                # is done by simply updating `uuid_to_storage` to the new custom storage objects).
                new_uuid_to_storage = self.training_pipeline.current_stage_storage
                storage_to_initialize = [
                    s
                    for uuid, s in new_uuid_to_storage.items()
                    if uuid not in uuid_to_storage  # Don't initialize storage already in use
                ]
                self.initialize_storage_and_viz(
                    storage_to_initialize=storage_to_initialize,
                )
                uuid_to_storage = new_uuid_to_storage

                # Change engine attributes that depend on the current stage
                self.training_pipeline.current_stage.change_engine_attributes(self)

            already_saved_checkpoint = False

            if self.is_distributed:
                self.num_workers_done.set("done", str(0))
                self.num_workers_steps.set("steps", str(0))
                # Ensure all workers are done before incrementing num_workers_{steps, done}
                dist.barrier(
                    device_ids=(None if self.device == torch.device("cpu") else [self.device.index])
                )

            self.former_steps = self.step_count
            former_storage_experiences = {
                k: v.total_experiences
                for k, v in self.training_pipeline.current_stage_storage.items()
            }

            if self.training_pipeline.rollout_storage_uuid is None:
                # In this case we're not expecting to collect storage experiences, i.e. everything
                # will be off-policy.

                # self.step_count is normally updated by the `self.collect_step_across_all_task_samplers`
                # call below, but since we're not collecting onpolicy experiences, we need to update
                # it here. The step count here is now just effectively a count of the number of times
                # we've called `compute_losses_track_them_and_backprop` below.
                self.step_count += 1

                before_update_info = dict(
                    next_value=None,
                    next_c_value=None,
                    use_gae=cur_stage_training_settings.use_gae,
                    gamma=cur_stage_training_settings.gamma,
                    tau=cur_stage_training_settings.gae_lambda,
                    adv_stats_callback=self.advantage_stats,
                )
            else:
                vector_tasks_already_restarted = False
                step = -1

                while step < cur_stage_training_settings.num_steps - 1:
                    step += 1

                    try:
                        num_paused = self.collect_step_across_all_task_samplers(
                            rollout_storage_uuid=self.training_pipeline.rollout_storage_uuid,
                            uuid_to_storage=uuid_to_storage,
                        )
                    except (TimeoutError, EOFError) as e:
                        if (not self.try_restart_after_task_error) or self.mode != TRAIN_MODE_STR:
                            # Apparently you can just call `raise` here and doing so will just raise the exception as though
                            # it was not caught (so the stacktrace isn't messed up)
                            raise
                        elif vector_tasks_already_restarted:
                            raise RuntimeError(
                                f"[{self.mode} worker {self.worker_id}] `vector_tasks` has timed out twice in the same"
                                f" rollout. This suggests that this error was not recoverable. Timeout exception:\n{traceback.format_exc()}"
                            )
                        else:
                            get_logger().warning(
                                f"[{self.mode} worker {self.worker_id}] `vector_tasks` appears to have crashed during"
                                f" training due to an {type(e).__name__} error. You have set"
                                f" `try_restart_after_task_error` to `True` so we will attempt to restart these tasks from"
                                f" the beginning. USE THIS FEATURE AT YOUR OWN"
                                f" RISK. Exception:\n{traceback.format_exc()}."
                            )
                            self.vector_tasks.close()
                            self._vector_tasks = None

                            vector_tasks_already_restarted = True
                            for storage in self.training_pipeline.current_stage_storage.values():
                                storage.after_updates()
                            self.initialize_storage_and_viz(
                                storage_to_initialize=cast(
                                    List[ExperienceStorage],
                                    list(uuid_to_storage.values()),
                                )
                            )
                            step = -1
                            continue

                    # A more informative error message should already have been thrown in be given in
                    # `collect_step_across_all_task_samplers` if `num_paused != 0` here but this serves
                    # as a sanity check.
                    assert num_paused == 0

                    if self.is_distributed:
                        # Preempt stragglers
                        # Each worker will stop collecting steps for the current rollout whenever a
                        # 100 * distributed_preemption_threshold percentage of workers are finished collecting their
                        # rollout steps, and we have collected at least 25% but less than 90% of the steps.
                        num_done = int(self.num_workers_done.get("done"))
                        if (
                            num_done > self.distributed_preemption_threshold * self.num_workers
                            and 0.25 * cur_stage_training_settings.num_steps
                            <= step
                            < 0.9 * cur_stage_training_settings.num_steps
                        ):
                            get_logger().debug(
                                f"[{self.mode} worker {self.worker_id}] Preempted after {step}"
                                f" steps (out of {cur_stage_training_settings.num_steps})"
                                f" with {num_done} workers done"
                            )
                            break

                with torch.no_grad():
                    actor_critic_output, _ = self.actor_critic(
                        **rollout_storage.agent_input_for_next_step()
                    )

                self.training_pipeline.rollout_count += 1

                if self.is_distributed:
                    # Mark that a worker is done collecting experience
                    self.num_workers_done.add("done", 1)
                    self.num_workers_steps.add("steps", self.step_count - self.former_steps)

                    # Ensure all workers are done before updating step counter
                    dist.barrier(
                        device_ids=(
                            None if self.device == torch.device("cpu") else [self.device.index]
                        )
                    )

                    ndone = int(self.num_workers_done.get("done"))
                    assert (
                        ndone == self.num_workers
                    ), f"# workers done {ndone} != # workers {self.num_workers}"

                    # get the actual step_count
                    self.step_count = int(self.num_workers_steps.get("steps")) + self.former_steps

                before_update_info = dict(
                    next_value=actor_critic_output.values.detach(),
                    next_c_value=actor_critic_output.c_values.detach(),
                    use_gae=cur_stage_training_settings.use_gae,
                    gamma=cur_stage_training_settings.gamma,
                    tau=cur_stage_training_settings.gae_lambda,
                    adv_stats_callback=self.advantage_stats,
                )

            # Prepare storage for iteration during updates
            for storage in self.training_pipeline.current_stage_storage.values():
                storage.before_updates(**before_update_info)

            for sc in self.training_pipeline.current_stage.stage_components:
                component_storage = uuid_to_storage[sc.storage_uuid]

                self.compute_losses_track_them_and_backprop(
                    stage=self.training_pipeline.current_stage,
                    stage_component=sc,
                    storage=component_storage,
                )

            for storage in self.training_pipeline.current_stage_storage.values():
                storage.after_updates()

            # We update the storage step counts saved in
            # `self.training_pipeline.current_stage.storage_uuid_to_steps_taken_in_stage` here rather than with
            # `self.steps` above because some storage step counts may only change after the update calls above.
            # This may seem a bit weird but consider a storage that corresponds to a fixed dataset
            # used for imitation learning. For such a dataset, the "steps" will only increase as
            # new batches are sampled during update calls.
            # Note: We don't need to sort the keys below to ensure that distributed updates happen correctly
            #   as `self.training_pipeline.current_stage_storage` is an ordered `dict`.
            # First we calculate the change in counts (possibly aggregating across devices)
            change_in_storage_experiences = {}
            for k in sorted(self.training_pipeline.current_stage_storage.keys()):
                delta = (
                    self.training_pipeline.current_stage_storage[k].total_experiences
                    - former_storage_experiences[k]
                )
                assert delta >= 0
                change_in_storage_experiences[k] = self.distributed_weighted_sum(
                    to_share=delta, weight=1
                )

            # Then we update `self.training_pipeline.current_stage.storage_uuid_to_steps_taken_in_stage` with the above
            # computed changes.
            for storage_uuid, delta in change_in_storage_experiences.items():
                self.training_pipeline.current_stage.storage_uuid_to_steps_taken_in_stage[
                    storage_uuid
                ] += delta

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch=self.training_pipeline.total_steps)

            # Here we handle saving a checkpoint every `save_interval` steps, saving after
            # a pipeline stage completes is controlled above
            checkpoint_file_name = None
            if should_save_checkpoints and (
                self.training_pipeline.total_steps - self.last_save
                >= cur_stage_training_settings.save_interval
            ):
                checkpoint_file_name = (
                    self._save_checkpoint_then_send_checkpoint_for_validation_and_update_last_save_counter()
                )
                already_saved_checkpoint = True

            if (
                self.training_pipeline.total_steps - self.last_log >= self.log_interval
                or self.training_pipeline.current_stage.is_complete
            ):
                self.aggregate_and_send_logging_package(
                    tracking_info_list=self.tracking_info_list,
                    checkpoint_file_name=checkpoint_file_name,
                )
                self.tracking_info_list.clear()
                self.last_log = self.training_pipeline.total_steps

            if (cur_stage_training_settings.advance_scene_rollout_period is not None) and (
                self.training_pipeline.rollout_count
                % cur_stage_training_settings.advance_scene_rollout_period
                == 0
            ):
                get_logger().info(
                    f"[{self.mode} worker {self.worker_id}] Force advance"
                    f" tasks with {self.training_pipeline.rollout_count} rollouts"
                )
                self.vector_tasks.next_task(force_advance_scene=True)
                self.initialize_storage_and_viz(
                    storage_to_initialize=cast(
                        List[ExperienceStorage], list(uuid_to_storage.values())
                    )
                )


class SafeOnPolicyRunner(OnPolicyRunner):
    def start_train(
        self,
        checkpoint: Optional[str] = None,
        restart_pipeline: bool = False,
        max_sampler_processes_per_worker: Optional[int] = None,
        save_ckpt_after_every_pipeline_stage: bool = True,
        collect_valid_results: bool = False,
        valid_on_initial_weights: bool = False,
        try_restart_after_task_error: bool = False,
        save_ckpt_at_every_host: bool = False,
        cost_limit: float = None,
    ):
        assert cost_limit is not None, "cost_limit must be set"
        self._initialize_start_train_or_start_test()

        self._collect_valid_results = collect_valid_results

        if not self.disable_config_saving:
            self.save_project_state()

        devices = self.worker_devices(TRAIN_MODE_STR)
        num_workers = len(devices)

        # Be extra careful to ensure that all models start
        # with the same initializations.
        set_seed(self.seed)
        initial_model_state_dict = self.config.create_model(
            sensor_preprocessor_graph=MachineParams.instance_from(
                self.config.machine_params(self.mode)
            ).sensor_preprocessor_graph
        ).state_dict()

        distributed_port = 0 if num_workers == 1 else self.get_port()

        if (
            num_workers > 1
            and "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ
            and "NCCL_BLOCKING_WAIT" not in os.environ
        ):
            # This ensures the NCCL distributed backend will throw errors
            # if we timeout at a call to `barrier()`
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

        worker_ids = self.local_worker_ids(TRAIN_MODE_STR)

        if checkpoint is not None:
            if checkpoint[:8] == "wandb://":
                ckpt_dir = "/tmp/wandb_ckpts"
                os.makedirs(ckpt_dir, exist_ok=True)
                checkpoint = download_checkpoint_from_wandb(
                    checkpoint, ckpt_dir, only_allow_one_ckpt=True
                )

        model_hash = None
        for trainer_id in worker_ids:
            training_kwargs = dict(
                id=trainer_id,
                checkpoint=checkpoint,
                restart_pipeline=restart_pipeline,
                experiment_name=self.experiment_name,
                config=self.config,
                callback_sensors=self._get_callback_sensors,
                results_queue=self.queues["results"],
                checkpoints_queue=(self.queues["checkpoints"] if self.running_validation else None),
                checkpoints_dir=self.checkpoint_dir(),
                seed=self.seed,
                deterministic_cudnn=self.deterministic_cudnn,
                mp_ctx=self.mp_ctx,
                num_workers=num_workers,
                device=devices[trainer_id],
                distributed_ip=self.distributed_ip_and_port.split(":")[0],
                distributed_port=distributed_port,
                max_sampler_processes_per_worker=max_sampler_processes_per_worker,
                save_ckpt_after_every_pipeline_stage=save_ckpt_after_every_pipeline_stage,
                initial_model_state_dict=(
                    initial_model_state_dict if model_hash is None else model_hash
                ),
                first_local_worker_id=worker_ids[0],
                distributed_preemption_threshold=self.distributed_preemption_threshold,
                valid_on_initial_weights=valid_on_initial_weights,
                try_restart_after_task_error=try_restart_after_task_error,
                save_ckpt_at_every_host=save_ckpt_at_every_host,
                cost_limit=cost_limit,
            )
            train: BaseProcess = self.mp_ctx.Process(
                target=self.train_loop,
                kwargs=training_kwargs,
            )
            try:
                train.start()
            except (ValueError, OSError, ConnectionRefusedError, EOFError) as e:
                # If the `initial_model_state_dict` is too large we sometimes
                # run into errors passing it with multiprocessing. In such cases
                # we instead hash the state_dict and confirm, in each engine worker, that
                # this hash equals the model the engine worker instantiates.
                if (
                    (isinstance(e, ValueError) and e.args[0] == "too many fds")
                    or (isinstance(e, OSError) and e.errno == 22)
                    or (isinstance(e, ConnectionRefusedError) and e.errno == 111)
                    or isinstance(e, EOFError)
                ):
                    model_hash = md5_hash_of_state_dict(initial_model_state_dict)
                    training_kwargs["initial_model_state_dict"] = model_hash
                    train = self.mp_ctx.Process(
                        target=self.train_loop,
                        kwargs=training_kwargs,
                    )
                    train.start()
                else:
                    raise e

            self.processes[TRAIN_MODE_STR].append(train)

        get_logger().info(f"Started {len(self.processes[TRAIN_MODE_STR])} train processes")

        # Validation
        if self.running_validation:
            device = self.worker_devices(VALID_MODE_STR)[0]
            self.init_visualizer(VALID_MODE_STR)
            valid: BaseProcess = self.mp_ctx.Process(
                target=self.valid_loop,
                args=(0,),
                kwargs=dict(
                    config=self.config,
                    callback_sensors=self._get_callback_sensors,
                    results_queue=self.queues["results"],
                    checkpoints_queue=self.queues["checkpoints"],
                    seed=12345,  # TODO allow same order for randomly sampled tasks? Is this any useful anyway?
                    deterministic_cudnn=self.deterministic_cudnn,
                    deterministic_agents=self.deterministic_agents,
                    mp_ctx=self.mp_ctx,
                    device=device,
                    max_sampler_processes_per_worker=max_sampler_processes_per_worker,
                ),
            )
            valid.start()
            self.processes[VALID_MODE_STR].append(valid)

            get_logger().info(f"Started {len(self.processes[VALID_MODE_STR])} valid processes")
        else:
            get_logger().info("No processes allocated to validation, no validation will be run.")

        metrics_file_template: Optional[str] = None

        if self._collect_valid_results:
            metrics_dir = self.metric_path(self.local_start_time_str)
            os.makedirs(metrics_dir, exist_ok=True)
            suffix = f"__valid_{self.local_start_time_str}"
            metrics_file_template = os.path.join(
                metrics_dir, "metrics" + suffix + "{:012d}.json"
            )  # template for training steps

            get_logger().info(f"Saving valid metrics with template {metrics_file_template}")

            # Check output file can be written
            with open(metrics_file_template.format(0), "w") as f:
                json.dump([], f, indent=4, sort_keys=True, cls=NumpyJSONEncoder)

        valid_results = self.log_and_close(
            start_time_str=self.local_start_time_str,
            nworkers=len(worker_ids),  # TODO num_workers once we forward metrics,
            metrics_file=metrics_file_template,
        )

        if not self._collect_valid_results:
            return self.local_start_time_str
        else:
            return self.local_start_time_str, valid_results

    @staticmethod
    def init_process(mode: str, id: int, to_close_on_termination: SafeOnPolicyRLEngine):
        ptitle(f"{mode}-{id}")

        def create_handler(termination_type: str):
            def handler(_signo, _frame):
                prefix = f"{termination_type} signal sent to worker {mode}-{id}."
                if to_close_on_termination.is_closed:
                    get_logger().info(f"{prefix} Worker {mode}-{id} is already closed, exiting.")
                    sys.exit(0)
                elif not to_close_on_termination.is_closing:
                    get_logger().info(f"{prefix} Forcing worker {mode}-{id} to close and exiting.")
                    # noinspection PyBroadException
                    try:
                        to_close_on_termination.close(True)
                    except Exception:
                        get_logger().error(
                            f"Error occurred when closing the RL engine used by work {mode}-{id}."
                            f" We cannot recover from this and will simply exit. The exception:\n"
                            f"{traceback.format_exc()}"
                        )
                        sys.exit(1)
                    sys.exit(0)
                else:
                    get_logger().info(
                        f"{prefix} Worker {mode}-{id} is already closing, ignoring this signal."
                    )

            return handler

        signal.signal(signal.SIGTERM, create_handler("Termination"))
        signal.signal(signal.SIGINT, create_handler("Interrupt"))

    @staticmethod
    def train_loop(
        id: int = 0,
        checkpoint: Optional[str] = None,
        restart_pipeline: bool = False,
        valid_on_initial_weights: bool = False,
        *engine_args,
        **engine_kwargs,
    ):
        engine_kwargs["mode"] = TRAIN_MODE_STR
        engine_kwargs["worker_id"] = id
        engine_kwargs_for_print = {
            k: (v if k != "initial_model_state_dict" else "[SUPPRESSED]")
            for k, v in engine_kwargs.items()
        }
        get_logger().info(f"train {id} args {engine_kwargs_for_print}")

        trainer: SafeOnPolicyTrainer = SafeOnPolicyRunner.init_worker(
            engine_class=SafeOnPolicyTrainer, args=engine_args, kwargs=engine_kwargs
        )
        if trainer is not None:
            SafeOnPolicyRunner.init_process("Train", id, to_close_on_termination=trainer)
            trainer.train(
                checkpoint_file_name=checkpoint,
                restart_pipeline=restart_pipeline,
                valid_on_initial_weights=valid_on_initial_weights,
            )


class SafeOnPolicyTrainer(SafeOnPolicyRLEngine, OnPolicyTrainer):
    def __init__(
        self,
        experiment_name: str,
        config: ExperimentConfig,
        results_queue: mp.Queue,
        checkpoints_queue: Optional[mp.Queue],
        checkpoints_dir: str = "",
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        mp_ctx: Optional[BaseContext] = None,
        worker_id: int = 0,
        num_workers: int = 1,
        device: Union[str, torch.device, int] = "cpu",
        distributed_ip: str = "127.0.0.1",
        distributed_port: int = 0,
        deterministic_agents: bool = False,
        distributed_preemption_threshold: float = 0.7,
        max_sampler_processes_per_worker: Optional[int] = None,
        save_ckpt_after_every_pipeline_stage: bool = True,
        first_local_worker_id: int = 0,
        save_ckpt_at_every_host: bool = False,
        **kwargs,
    ):
        kwargs["mode"] = TRAIN_MODE_STR
        super().__init__(
            experiment_name=experiment_name,
            config=config,
            results_queue=results_queue,
            checkpoints_queue=checkpoints_queue,
            checkpoints_dir=checkpoints_dir,
            seed=seed,
            deterministic_cudnn=deterministic_cudnn,
            mp_ctx=mp_ctx,
            worker_id=worker_id,
            num_workers=num_workers,
            device=device,
            distributed_ip=distributed_ip,
            distributed_port=distributed_port,
            deterministic_agents=deterministic_agents,
            max_sampler_processes_per_worker=max_sampler_processes_per_worker,
            **kwargs,
        )

        self.save_ckpt_after_every_pipeline_stage = save_ckpt_after_every_pipeline_stage

        self.actor_critic.train()

        self.training_pipeline: TrainingPipeline = config.training_pipeline()

        if self.num_workers != 1:
            # Ensure that we're only using early stopping criterions in the non-distributed setting.
            if any(
                stage.early_stopping_criterion is not None
                for stage in self.training_pipeline.pipeline_stages
            ):
                raise NotImplementedError(
                    "Early stopping criterions are currently only allowed when using a single training worker, i.e."
                    " no distributed (multi-GPU) training. If this is a feature you'd like please create an issue"
                    " at https://github.com/allenai/allenact/issues or (even better) create a pull request with this "
                    " feature and we'll be happy to review it."
                )

        self.optimizer: optim.optimizer.Optimizer = self.training_pipeline.optimizer_builder(
            params=[p for p in self.actor_critic.parameters() if p.requires_grad]
        )
        self._lagrange: Lagrange = Lagrange(
            **{
                "cost_limit": kwargs["cost_limit"],
                "lagrangian_multiplier_init": 0.001,
                "lambda_lr": 0.035,
                "lambda_optimizer": "Adam",
            }
        )

        # noinspection PyProtectedMember
        self.lr_scheduler: Optional[_LRScheduler] = None
        if self.training_pipeline.lr_scheduler_builder is not None:
            self.lr_scheduler = self.training_pipeline.lr_scheduler_builder(
                optimizer=self.optimizer
            )

        if self.is_distributed:
            # Tracks how many workers have finished their rollout
            self.num_workers_done = torch.distributed.PrefixStore(  # type:ignore
                "num_workers_done", self.store
            )
            # Tracks the number of steps taken by each worker in current rollout
            self.num_workers_steps = torch.distributed.PrefixStore(  # type:ignore
                "num_workers_steps", self.store
            )
            self.distributed_preemption_threshold = distributed_preemption_threshold
            # Flag for finished worker in current epoch
            self.offpolicy_epoch_done = torch.distributed.PrefixStore(  # type:ignore
                "offpolicy_epoch_done", self.store
            )
            # Flag for finished worker in current epoch with custom component
            self.insufficient_data_for_update = torch.distributed.PrefixStore(  # type:ignore
                "insufficient_data_for_update", self.store
            )
        else:
            self.num_workers_done = None
            self.num_workers_steps = None
            self.distributed_preemption_threshold = 1.0
            self.offpolicy_epoch_done = None

        # Keeping track of training state
        self.former_steps: Optional[int] = None
        self.last_log: Optional[int] = None
        self.last_save: Optional[int] = None
        # The `self._last_aggregated_train_task_metrics` attribute defined
        # below is used for early stopping criterion computations
        self._last_aggregated_train_task_metrics: ScalarMeanTracker = ScalarMeanTracker()

        self.first_local_worker_id = first_local_worker_id
        self.save_ckpt_at_every_host = save_ckpt_at_every_host


# Re-export SafeRL-enabled classes as the default
RolloutBlockStorage = SafeRolloutBlockStorage
OnPolicyRunner = SafeOnPolicyRunner
