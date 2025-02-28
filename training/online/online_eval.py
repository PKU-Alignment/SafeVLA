import argparse
import datetime
import os
# from online_evaluation.local_logging_utils import LocalWandb
import wandb
import torch

from architecture.models.allenact_transformer_models.inference_agent import InferenceAgentVIDA
from architecture.allenact_preprocessors.dino_preprocessors import DinoViTPreprocessor

from training.online.dinov2_vits_tsfm_rgb_augment_objectnav import (
    DinoV2ViTSTSFMObjectNav,
    DinoV2ViTSTSFMObjectNavParams,
)
from online_evaluation.online_evaluator import OnlineEvaluatorManager
from safety_gymnasium.tasks.safe_vla import REGISTERED_TASKS
from training.online.dataset_mixtures import get_mixture_by_name

img_encoder_type = {
    "DinoV2": {
        "mean": DinoViTPreprocessor.DINO_RGB_MEANS,
        "std": DinoViTPreprocessor.DINO_RGB_STDS,
    },
}

model_config_type = {
    "InferenceDINOv2ViTSLLAMATxTxObjectNavDist": DinoV2ViTSTSFMObjectNav,
}

model_config_params = {
    "InferenceDINOv2ViTSLLAMATxTxObjectNavDist": DinoV2ViTSTSFMObjectNavParams,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Online evaluation")
    parser.add_argument(
        "--model_config", default="InferenceDINOv2ViTSLLAMATxTxObjectNavDist", type=str
    )
    parser.add_argument("--training_tag", type=str)
    parser.add_argument("--wandb_project_name", type=str, default="safevla_eval")
    parser.add_argument("--wandb_entity_name", type=str, default="rl_train")
    parser.add_argument("--training_run_id", type=str)
    parser.add_argument("--ckpt_path", default="/net/nfs2.prior/checkpoints/")
    parser.add_argument("--max_eps_len", default=-1, type=int)
    parser.add_argument("--eval_set_size", default=None, type=int)
    parser.add_argument("--greedy", action="store_true", default=False)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--test_augmentation", action="store_true", default=False)
    parser.add_argument("--eval_subset", default="minival", help="options: val, minival, train")
    parser.add_argument("--dataset_type", default="object_nav_v0.3")
    parser.add_argument("--task_type", default="ObjectNavType")
    parser.add_argument("--img_encoder_type", default="DinoV2")
    parser.add_argument("--dataset_path", default="/data/datasets")
    parser.add_argument("--output_basedir", default="/data/results/online_evaluation")
    parser.add_argument("--house_set", default="objaverse", help="procthor or objaverse")
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--extra_tag", default="")
    parser.add_argument("--max_restart_times", type=int, default=1)
    parser.add_argument("--benchmark_revision", default="chores-small")
    parser.add_argument("--det_type", default="gt")
    parser.add_argument("--gpu_devices", nargs="+", default=[], type=int)
    parser.add_argument("--ignore_text_goal", action="store_true", default=False)
    parser.add_argument(
        "--input_sensors",
        nargs="+",
        default=["raw_navigation_camera"],
    )

    args = parser.parse_args()

    if len(args.gpu_devices) == 1 and args.gpu_devices[0] == -1:
        args.gpu_devices = None
    elif len(args.gpu_devices) == 0:
        # Get all the available GPUS
        args.gpu_devices = [i for i in range(torch.cuda.device_count())]

    return args


def load_model_configs(
    exp_config_type,
    device,
    ckpt_path,
    img_preprocessor_mean_std,
    greedy_sampling,
    test_augmentation,
):
    exp_config = exp_config_type(num_train_processes=1, train_gpu_ids=[device])

    agent = InferenceAgentVIDA.from_experiment_config(
        exp_config=exp_config,
        device=device,
        mode="test",
    )
    agent.img_encoder_rgb_mean = img_preprocessor_mean_std["mean"]
    agent.img_encoder_rgb_std = img_preprocessor_mean_std["std"]
    agent.greedy_sampling = greedy_sampling
    agent.test_augmentation = test_augmentation
    agent.augmentations = None

    agent.actor_critic.load_state_dict(
        torch.load(ckpt_path, map_location=device)["model_state_dict"]
    )


    agent.steps_before_rollout_refresh = 10000

    agent.reset()

    return agent


def get_eval_run_name(args):    
    # exp_name = ["OnlineEval"]

    # if args.extra_tag != "":
    #     exp_name.append(f"{args.extra_tag}")
    exp_name = args.task_type
    # exp_name.extend(s\
    #     [
    #         f"training_run_id={args.training_run_id}",
    #         f"eval_dataset={args.dataset_type}",
    #         f"eval_subset={args.eval_subset}",
    #         f"shuffle={args.shuffle}",
    #         f"greedy_sampling={args.greedy}",
    #         f"test_augmentation={args.test_augmentation}",
    #     ]
    # )

    return exp_name


def main(args):
    eval_run_name = get_eval_run_name(args)
    exp_base_dir = os.path.join(args.output_basedir, eval_run_name)
    exp_dir = os.path.join(exp_base_dir, datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f"))
    os.makedirs(exp_dir, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity_name,
        name=eval_run_name,
        dir=os.path.join(exp_dir, "wandb"),
    )
    class WandbWrapper:
        def __init__(self, run):
            self.run = run
            self.Table = wandb.Table 
            
        def log(self, *args, **kwargs):
            return self.run.log(*args, **kwargs)
            
    preset_wandb = WandbWrapper(run)

    if args.task_type not in REGISTERED_TASKS:
        list_of_tasks = get_mixture_by_name(args.task_type)
        assert args.eval_subset == "minival"
        dataset_type = ""
        dataset_path = ""
    else:
        list_of_tasks = [args.task_type]
        dataset_type = args.dataset_type
        dataset_path = args.dataset_path

    devices = ["cpu"]
    if args.gpu_devices is not None and len(args.gpu_devices) > 0:
        devices = args.gpu_devices

    evaluator = OnlineEvaluatorManager(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        max_eps_len=args.max_eps_len,
        eval_set_size=args.eval_set_size,
        eval_subset=args.eval_subset,
        shuffle=args.shuffle,
        gpu_devices=devices,
        outdir=exp_dir,
        list_of_tasks=list_of_tasks,
        input_sensors=args.input_sensors,
        house_set=args.house_set,
        num_workers=args.num_workers,
        preset_wandb=preset_wandb,
        benchmark_revision=args.benchmark_revision,
        det_type=args.det_type,
        extra_tag=args.extra_tag,
        max_restart_times=args.max_restart_times,
    )

    params = model_config_params[args.model_config]()
    params.num_train_processes = 0
    if args.ignore_text_goal:
        params.use_text_goal = False
    if any("box" in s for s in args.input_sensors):
        params.use_bbox = True
    else:
        params.use_bbox = False

    agent_input = dict(
        exp_config_type=model_config_type[args.model_config],
        params=params,
        img_encoder_rgb_mean=img_encoder_type[args.img_encoder_type]["mean"],
        img_encoder_rgb_std=img_encoder_type[args.img_encoder_type]["std"],
        greedy_sampling=args.greedy,
        test_augmentation=args.test_augmentation,
        ckpt_path=args.ckpt_path,
    )
    evaluator.evaluate(InferenceAgentVIDA, agent_input)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    # os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
    args = parse_args()
    main(args)
