import argparse
import datetime
import os

import torch
import wandb
from allenact.utils.misc_utils import str2bool

from architecture.models.transformer_models import REGISTERED_MODELS
from online_evaluation.local_logging_utils import LoadLocalWandb, LocalWandb
from online_evaluation.online_evaluator import OnlineEvaluatorManager
from tasks import REGISTERED_TASKS
from training.offline.dataset_mixtures import get_mixture_by_name


def parse_args():
    parser = argparse.ArgumentParser(description="Online evaluation")
    parser.add_argument("--training_run_id", type=str)
    parser.add_argument("--ckptStep", default=None, type=int)
    parser.add_argument("--max_eps_len", default=-1, type=int)
    parser.add_argument("--eval_set_size", default=200, type=int)
    parser.add_argument("--sampling", default="sample")
    parser.add_argument("--gpu_devices", nargs="+", default=[], type=int)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--test_augmentation", action="store_true", default=False)
    parser.add_argument("--skip_done", action="store_true", default=False)
    parser.add_argument("--eval_subset", default="minival", help="options: val, minival, train")
    parser.add_argument("--dataset_type", default="")
    parser.add_argument("--task_type", default="")
    parser.add_argument("--det_type", default="gt", help="gt or detic", choices=["gt", "detic"])
    parser.add_argument("--house_set", default="procthor", help="procthor or objaverse")
    parser.add_argument("--dataset_path", default="/data/datasets")
    parser.add_argument("--output_basedir", default="/data/results/online_evaluation")
    parser.add_argument("--local_checkpoint_dir", default="/data/results/checkpoint_dir")
    parser.add_argument("--extra_tag", default="")
    parser.add_argument("--real_evaluation", action="store_true", default=False)
    parser.add_argument("--stretch_host", default="172.16.121.205")
    parser.add_argument("--benchmark_revision", default="chores-small")
    parser.add_argument("--wandb_logging", default=True, type=str2bool)
    parser.add_argument("--wandb_project_name", default="", type=str)
    parser.add_argument("--wandb_entity_name", default="", type=str)
    parser.add_argument(
        "--input_sensors",
        nargs="+",
        default=["raw_navigation_camera", "raw_manipulation_camera"],
    )
    parser.add_argument("--model_version_override", default="auto")
    parser.add_argument("--total_num_videos", type=int, default=8200)

    args = parser.parse_args()

    if len(args.gpu_devices) == 1 and args.gpu_devices[0] == -1:
        args.gpu_devices = None
    elif len(args.gpu_devices) == 0:
        # Get all the available GPUS
        args.gpu_devices = [i for i in range(torch.cuda.device_count())]

    if args.wandb_logging:
        assert args.wandb_project_name != ""
        assert args.wandb_entity_name != ""

    return args


def get_eval_run_name(args):
    exp_name = ["OnlineEval-revision-{}".format(args.benchmark_revision)]

    if args.extra_tag != "":
        exp_name.append(f"extra_tag={args.extra_tag}")

    if args.ckptStep is not None:
        exp_name.append(f"ckptStep={args.ckptStep}")

    exp_name.extend(
        [
            f"training_run_id={args.training_run_id}",
            f"eval_dataset={args.dataset_type}",
            f"eval_subset={args.eval_subset}",
            f"shuffle={args.shuffle}",
            f"sampling={args.sampling}",
        ]
    )

    return "-".join(exp_name)


def main(args):
    gpu_devices = ["cpu"]
    if args.gpu_devices is not None and len(args.gpu_devices) > 0:
        gpu_devices = [int(device) for device in args.gpu_devices]

    if args.wandb_logging:
        assert (
            args.wandb_entity_name != "" and args.wandb_project_name != ""
        ), "wandb_entity_name and wandb_project_name must be provided"
        api = wandb.Api()

        run = api.run(f"{args.wandb_entity_name}/{args.wandb_project_name}/{args.training_run_id}")
    else:
        run = LoadLocalWandb(run_id=args.training_run_id, save_dir=args.local_checkpoint_dir)
    # run = LoadLocalWandb(run_id=args.training_run_id, save_dir=args.local_checkpoint_dir)

    training_run_name = run.config["exp_name"]
    eval_run_name = get_eval_run_name(args)
    exp_base_dir = os.path.join(args.output_basedir, eval_run_name)
    ckpt_dir = os.path.join(exp_base_dir, "ckpts")
    exp_dir = os.path.join(exp_base_dir, datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f"))
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    if args.wandb_logging:
        if args.ckptStep is None:
            raise ValueError("ckptStep is None")
        assert (
            args.wandb_entity_name != "" and args.wandb_project_name != ""
        ), "wandb_entity_name and wandb_project_name must be provided"
        ckpt_fn = f"{args.wandb_entity_name}/{args.wandb_project_name}/ckpt-{args.training_run_id}-{args.ckptStep}:latest"
        artifact = api.artifact(ckpt_fn)
        artifact.download(ckpt_dir)
        ckpt_pth = os.path.join(ckpt_dir, "model.ckpt")
    else:
        ckpt_pth = run.get_checkpoint(ckpt_step=args.ckptStep)
    # ckpt_pth = run.get_checkpoint(ckpt_step=args.ckptStep)

    model = run.config["model"]
    model_input_sensors = run.config["input_sensors"]
    if args.input_sensors is not None:
        # some sensors (e.g rooms_seen, room_current_seen) that are need to create model
        # are self-predicted and may not be provided to the agent as input
        assert set(args.input_sensors).issubset(
            set(model_input_sensors)
        ), f"{set(args.input_sensors)} is not a subset of {set(model_input_sensors)}"

    model_version = run.config["model_version"]

    if args.model_version_override != "auto":
        print(f"Enforcing model_version {args.model_version_override}")
        model_version = args.model_version_override

    loss = run.config["loss"]

    agent_class = REGISTERED_MODELS[model]
    agent_input = dict(
        model_version=model_version,
        input_sensors=model_input_sensors,
        loss=loss,
        sampling=args.sampling,
        data_augmentation=args.test_augmentation,
        ckpt_pth=ckpt_pth,
    )


    assert not args.wandb_logging or (
        args.wandb_entity_name != "" and args.wandb_project_name != ""
    ), "wandb_entity_name and wandb_project_name must be provided"
    if args.wandb_logging:
        preset_wandb = make_wandb(
            wandb_project=args.wandb_project_name,
            wandb_entity=args.wandb_entity_name,
            wandb_name=eval_run_name,
            wandb_directory=os.path.join(exp_dir, "wandb"),
        )
    else:
        preset_wandb = LocalWandb(
            project=args.wandb_project_name,
            entity=args.wandb_entity_name,
            name=eval_run_name,
            save_dir=os.path.join(exp_dir, "wandb"),
        )

    if args.task_type not in REGISTERED_TASKS:
        list_of_tasks = get_mixture_by_name(args.task_type)
        assert args.eval_subset == "minival"
        dataset_type = ""
        dataset_path = ""
    else:
        list_of_tasks = [args.task_type]
        dataset_type = args.dataset_type
        dataset_path = args.dataset_path

    # eval_args = {
    #     "dataset_path": dataset_path,
    #     "dataset_type": dataset_type,
    #     "max_eps_len": args.max_eps_len,
    #     "eval_set_size": args.eval_set_size,
    #     "eval_subset": args.eval_subset,
    #     "shuffle": args.shuffle,
    #     "gpu_devices": gpu_devices,
    #     "outdir": exp_dir,
    #     "list_of_tasks": list_of_tasks,
    #     "input_sensors": args.input_sensors,
    #     "skip_done": args.skip_done,
    #     "house_set": args.house_set,
    #     "num_workers": args.num_workers,
    #     "preset_wandb": preset_wandb,
    #     "table_size": args.total_num_videos,
    #     "benchmark_revision": args.benchmark_revision,
    #     "det_type": args.det_type,
    # }
    devices = ["cpu"]
    if args.gpu_devices is not None and len(args.gpu_devices) > 0:
        # devices = [f"cuda:{device}" for device in args.gpu_devices]
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
        table_size=args.total_num_videos,
        benchmark_revision=args.benchmark_revision,
        det_type=args.det_type,
        # real_evaluation=args.real_evaluation,
        # stretch_host=args.stretch_host,
    )

    # evaluator = OnlineEvaluatorManager(**eval_args)
    evaluator.evaluate(agent_class, agent_input)


def make_wandb(wandb_project, wandb_entity, wandb_name, wandb_directory):
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name,
        dir=wandb_directory,
    )
    return wandb


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    args = parse_args()
    if args.wandb_logging:
        os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
    main(args)
