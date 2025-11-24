import abc
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Literal, Optional, Union

from allenact.algorithms.onpolicy_sync.runner import OnPolicyRunner, SaveDirFormat
from allenact.base_abstractions.experiment_config import ExperimentConfig


@dataclass
class OnPolicyRunnerMixin(abc.ABC):
    output_dir: str = "/root/results"
    save_dir_fmt: Literal["flat", "nested"] = "flat"
    seed: Optional[int] = None
    deterministic_cudnn: bool = False
    deterministic_agents: bool = False
    extra_tag: str = ""
    disable_tensorboard: bool = True
    disable_config_saving: bool = True
    distributed_ip_and_port: str = "127.0.0.1:0"
    machine_id: int = 0
    callbacks: str = ""
    cost_limit: float = None

    @abc.abstractmethod
    def get_config(self) -> ExperimentConfig:
        raise NotImplementedError

    def build_runner(self, mode=Literal["train", "test"]):
        return OnPolicyRunner(
            config=self.get_config(),
            output_dir=self.output_dir,
            save_dir_fmt=SaveDirFormat[self.save_dir_fmt.upper()],
            loaded_config_src_files=None,
            seed=self.seed,
            mode=mode,
            deterministic_cudnn=self.deterministic_cudnn,
            deterministic_agents=self.deterministic_agents,
            extra_tag=self.extra_tag,
            disable_tensorboard=self.disable_tensorboard,
            disable_config_saving=self.disable_config_saving,
            distributed_ip_and_port=self.distributed_ip_and_port,
            machine_id=self.machine_id,
            callbacks_paths=self.callbacks,
        )

    def train(
        self,
        checkpoint: Optional[
            str
        ] = "",
        restart_pipeline: bool = False,
        max_sampler_processes_per_worker: Optional[int] = None,
        collect_valid_results: bool = False,
        valid_on_initial_weights: bool = False,
        enable_crash_recovery: bool = False,
        save_ckpt_at_every_host: bool = False,
    ):
        runner = self.build_runner(mode="train")
        runner.start_train(
            checkpoint=checkpoint,
            restart_pipeline=restart_pipeline,
            max_sampler_processes_per_worker=max_sampler_processes_per_worker,
            collect_valid_results=collect_valid_results,
            valid_on_initial_weights=valid_on_initial_weights,
            try_restart_after_task_error=enable_crash_recovery,
            save_ckpt_at_every_host=save_ckpt_at_every_host,
            cost_limit=self.cost_limit,
        )

    def test(
        self,
        checkpoint: Optional[str] = None,
        infer_output_dir: str = False,
        approx_ckpt_step_interval: Optional[Union[float, int]] = None,
        max_sampler_processes_per_worker: Optional[int] = None,
        test_expert: bool = False,
    ):
        runner = self.build_runner(mode="test")
        runner.start_test(
            checkpoint_path_dir_or_pattern=checkpoint,
            infer_output_dir=infer_output_dir,
            approx_ckpt_step_interval=approx_ckpt_step_interval,
            max_sampler_processes_per_worker=max_sampler_processes_per_worker,
            inference_expert=test_expert,
        )

    def print_param(self, name):
        print(name, ": ", getattr(self, name))

    def get_params_dict(self) -> Dict[str, Any]:
        params = dict()
        for field in fields(self):
            params[field.name] = getattr(self, field.name)

        return params
