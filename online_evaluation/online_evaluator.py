import multiprocessing as mp
import os
import platform
import random
import time
from collections import defaultdict
from queue import Empty as EmptyQueueError
from typing import List, Dict
from datetime import datetime
import pandas as pd
import prior
import torch

from online_evaluation.online_evaluation_types_and_utils import (
    NormalizedEvalSample,
    EvalSample,
    eval_sample_to_normalized_eval_sample,
    map_hard_easy_objectnavtype_to_objectnavtype,
)
from utils.task_type_mapping_utils import inverse_map_task_type
from online_evaluation.online_evaluator_worker import (
    OnlineEvaluatorWorker,
    start_worker,
)
from tasks import REGISTERED_TASKS
from training.online.chores_dataset import ChoresDataset
from utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR

mp = mp.get_context("forkserver") if torch.cuda.is_available() else mp.get_context("spawn")
LOG_INTERMEDIATE_RESULTS = True

from utils.visualization_utils import VideoLogging
import signal
from contextlib import contextmanager
from utils.string_utils import json_templated_spec_to_dict


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def filter_task_specific_eps(tasks, task_type):
    filtered_tasks = {"train": [], "val": [], "test": []}

    # 3 if statements since tasks['train'] errors out if the training set is empty, so need to uuse tasks.train

    if tasks.train is not None:
        for i in range(len(tasks.train)):
            task = tasks.train[i]
            if task["task_type"] == task_type:
                filtered_tasks["train"].append(task)
    if tasks.val is not None:
        for i in range(len(tasks.val)):
            task = tasks.val[i]
            if task["task_type"] == task_type:
                filtered_tasks["val"].append(task)

    if tasks.test is not None:
        for i in range(len(tasks.test)):
            task = tasks.test[i]
            if task["task_type"] == task_type:
                filtered_tasks["train"].append(task)

    return filtered_tasks


class MetricAggregator:
    def __init__(self):
        self.sample_metrics = list()

    def update(self, metric):
        self.sample_metrics.append(metric)

    def aggregate(self):
        return sum(self.sample_metrics) / (len(self.sample_metrics) + 1e-10)

    def size(self):
        return len(self.sample_metrics)


def merge_current_metric_to_metric_aggregate_dict(metric_aggregate_dict, current_metric_value_dict):
    for k, v in current_metric_value_dict.items():
        if k not in metric_aggregate_dict:
            metric_aggregate_dict[k] = MetricAggregator()
        metric_aggregate_dict[k].update(v)


def log_results(
    wandb,
    all_workers_results,
    task_type,
    upload_per_task=True,
    upload_video=True,
    upload_per_synset=True,
):
    # task_path points out the episode's origin (i.e., which task, episode id, streaming id)

    table = None

    total_metric_aggregators_dict = {}
    contributing_workers = set()
    for result in all_workers_results:
        metric_values = result[0]["metrics"]
        tab_data = result[1]
        contributing_workers.add(result[0]["worker_id"])
        merge_current_metric_to_metric_aggregate_dict(total_metric_aggregators_dict, metric_values)

        if tab_data is not None:
            if table is None:
                columns = [k for k in tab_data.keys()]
                table = wandb.Table(columns=columns)
            row = []
            for k in columns:
                if k not in tab_data:
                    item_to_add = -1
                # elif k == "video_path":
                #     item_to_add = wandb.Video(tab_data["video_path"]) if upload_video else "None"
                # elif k == "topdown_view_path":
                #     item_to_add = (
                #         wandb.Image(tab_data["topdown_view_path"]) if upload_video else "None"
                #     )
                else:
                    item_to_add = tab_data[k]
                row.append(item_to_add)
            table.add_data(*row)
    if upload_video:
        wandb.log({f"VideoTable/{task_type}": table})

    all_objects = set(
        [o.split("/")[1] for o in total_metric_aggregators_dict.keys() if o.startswith("extra/")]
    )
    all_metrics = set(
        [
            o.split("/")[2]
            for o in total_metric_aggregators_dict.keys()
            if o.startswith("extra/")
            and len(o.split("/")) == 3  # one for extra one for object one for metric
        ]
    )
    if len(all_objects) > 0:
        columns = ["object_name"] + [x for x in all_metrics] + ["total_size"]
        final_result_table = wandb.Table(columns=columns)

        for o in all_objects:
            row = [o]
            size = 0
            for col in columns[1:-1]:
                key = f"extra/{o}/{col}"
                if key in total_metric_aggregators_dict:
                    row += [total_metric_aggregators_dict[key].aggregate()]
                    # Assume at least `eps_len` or `success` are available for all episodes, and include all tasks
                    size = max(size, total_metric_aggregators_dict[key].size())
                else:
                    row += [-1]
            row += [size]
            final_result_table.add_data(*row)

        if upload_per_task:
            wandb.log({f"PerObjectType/{task_type}": final_result_table})

    metrics_to_log = [
        k
        for k in total_metric_aggregators_dict.keys()
        if "extra/" not in k and "for_video_table" not in k
    ]
    metric_values = [total_metric_aggregators_dict[k].aggregate() for k in metrics_to_log]
    metrics_to_log += ["total_size"]
    metric_values += [total_metric_aggregators_dict[metrics_to_log[0]].size()]
    metrics_to_log += ["num_workers"]
    metric_values += [len(contributing_workers)]
    aggrgeated_result_metrics_table = wandb.Table(columns=metrics_to_log)
    aggrgeated_result_metrics_table.add_data(*metric_values)
    if upload_per_task:
        wandb.log({f"AggregatedResults/{task_type}": aggrgeated_result_metrics_table})
    return metrics_to_log, metric_values


class OnlineEvaluatorManager:
    def __init__(
        self,
        dataset_path="/data/datasets",
        dataset_type="object_nav_v0.3",
        max_eps_len=-1,
        eval_set_size=None,
        eval_subset="val",
        shuffle=True,
        seed=123,
        gpu_devices=None,
        outdir="/data/results/online_evaluation/OnlineEval-default",
        exist_ok=True,
        table_size=200,
        list_of_tasks=None,
        input_sensors=("raw_navigation_camera", "raw_manipulation_camera"),
        skip_done=False,
        house_set="procthor",
        num_workers=1,
        preset_wandb=None,
        benchmark_revision="chores-small",
        det_type=None,
        extra_tag="",
        prob_randomize_lighting=0,
        prob_randomize_materials=0,
        prob_randomize_colors=0,
    ):
        self.benchmark_revision = benchmark_revision

        assert benchmark_revision in [
            "chores-small",
            "chores-large",
            "15targets-p2p-roboapartment-Dec13",
            "astar-april2024",
            "500eval-april2024",
            "val-May2024",
        ], "other revisions are not supported yet"
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.max_eps_len = max_eps_len
        self.eval_set_size = eval_set_size
        self.eval_subset = eval_subset
        self.shuffle = shuffle
        self.seed = seed
        self.gpu_devices = gpu_devices
        self.controller_set = {i: None for i in range(num_workers)}
        self.num_workers = num_workers
        self.outdir = outdir
        self.exist_ok = exist_ok
        self.list_of_tasks = [] if list_of_tasks is None else list_of_tasks
        self.table_size = int(round(max(table_size / len(self.list_of_tasks), 1)))
        self.input_sensors = list(input_sensors)
        self.skip_done = skip_done
        self.house_set = house_set
        self.det_type = det_type
        self.extra_tag = extra_tag
        self.prob_randomize_lighting = prob_randomize_lighting
        self.prob_randomize_materials = prob_randomize_materials
        self.prob_randomize_colors = prob_randomize_colors
        os.makedirs(self.outdir, exist_ok=self.exist_ok)

        self.WorkerType = OnlineEvaluatorWorker

        self.logging_sensor = VideoLogging()
        self.wandb = preset_wandb

        if self.eval_subset in ["train", "val"]:
            assert len(self.list_of_tasks) == 1, "we do not support more than one for this option"
            self.eval_samples = self.load_full_eval_samples(self.list_of_tasks[0])
        else:
            self.eval_samples = self.load_minival_eval_samples(self.list_of_tasks)

        if self.house_set == "procthor":
            self.houses = self.load_procthor_houses()
        elif self.house_set == "objaverse":
            self.houses = self.load_objaverse_houses()
        elif self.house_set == "phone2proc":
            self.houses = self.load_phone2proc_houses()
        else:
            raise Exception("house_set not recognized", self.house_set)

        # eval_set_size has to be equal to the length of dataset if we are doing minival
        if self.eval_subset == "minival":
            for task in self.list_of_tasks:
                if self.eval_set_size != len(self.eval_samples[task]):
                    print(
                        " WARNING: You are expecting fewer samples but are minival has more samples",
                        "expected len eval",
                        self.eval_set_size,
                        "len minival",
                        len(self.eval_samples[task]),
                        "task",
                        task,
                    )

        self.num_ended_workers = 0
        self.num_tasks_in_queue = 0

    def load_minival_eval_samples(self, list_of_tasks) -> Dict[str, List[NormalizedEvalSample]]:
        all_task_samples = {}
        # Make a dictionary of tasks to list of samples
        for task in list_of_tasks:
            all_task_samples[task] = self.load_minival_eval_samples_per_task(task)
        return all_task_samples

    def load_minival_eval_samples_per_task(self, task_type: str) -> List[NormalizedEvalSample]:
        # TODO: This can be more efficient by reading first and then dividing.

        if self.benchmark_revision not in ["chores-small", "chores-large"]:
            EVAL_TASKS = prior.load_dataset(
                dataset="vida-benchmark",
                revision=self.benchmark_revision,
                task_types=[inverse_map_task_type(task_type)],
            )
        else:
            EVAL_TASKS = prior.load_dataset(
                dataset="spoc-data",
                entity="spoc-robot",
                revision=self.benchmark_revision,
                task_types=[inverse_map_task_type(task_type)],
            )

        samples: List[EvalSample] = EVAL_TASKS["val"]

        sample_ids = list(range(len(samples)))
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(sample_ids)

        if self.eval_set_size is not None:
            sample_ids = sample_ids[: self.eval_set_size]

        normalized_samples = [
            eval_sample_to_normalized_eval_sample(task_type=task_type, sample=samples[i], index=i)
            for i in range(len(samples))
        ]

        return [normalized_samples[i] for i in sample_ids]

    def load_full_eval_samples(self, task_type: str) -> Dict[str, List[NormalizedEvalSample]]:
        data_dir = os.path.join(self.dataset_path, self.dataset_type)
        samples = ChoresDataset(
            data_dir,
            subset=self.eval_subset,
            sliding_window=50,
            max_samples=self.eval_set_size,
            load_frames=False,
            input_sensors=self.input_sensors,
        )
        sample_ids = list(range(len(samples)))
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(sample_ids)

        if self.eval_set_size is not None:
            sample_ids = sample_ids[: self.eval_set_size]

        return {task_type: [samples[i] for i in sample_ids]}

    def load_procthor_houses(self):
        if self.eval_subset in ["val", "minival"]:
            return prior.load_dataset(
                dataset="spoc-data", entity="spoc-robot", revision="houses-test-val"
            )["val"]
        else:
            return prior.load_dataset(
                dataset="spoc-data", entity="spoc-robot", revision="houses-test-val"
            )[self.eval_subset]

    def load_phone2proc_houses(self):
        return prior.load_dataset(
            dataset="all-back-apartment-data",
        )["val"]

    def log_data_stat(self, task_type, samples):
        distribution_dict = {}
        for sample in samples:
            task_info = json_templated_spec_to_dict(sample["observations"]["templated_task_type"])

            for key in ["synsets", "room_types", "reference_synsets", "broad_synset_to_object_ids"]:
                if key in task_info:
                    distribution_dict.setdefault(key, {})
                    values = task_info[key]
                    for v in values:
                        distribution_dict[key][v] = distribution_dict[key].get(v, 0) + 1

            for key in [
                "room_type",
                "affordance",
                "reference_type",
                "rel_attribute",
                "num_rooms_in_house",
                "reference_synset",
            ]:
                if key in task_info:
                    distribution_dict.setdefault(key, {})
                    value = task_info[key]
                    if isinstance(value, list):
                        value = str(tuple(value))

                    distribution_dict[key][value] = distribution_dict[key].get(value, 0) + 1

        if distribution_dict == {}:
            if task_type not in ["RoomVisit"]:
                print("task_type", task_type, "does not have any data")

        if self.wandb is not None:
            for key in distribution_dict:
                table = self.wandb.Table(columns=["key", "count"])
                for k, v in distribution_dict[key].items():
                    table.add_data(k, v)
                self.wandb.log({f"DataStat/{task_type}/{key}": table})

    def load_objaverse_houses(self):
        if self.eval_subset in ["val", "minival"]:
            subset_to_load = "val"
        else:
            subset_to_load = self.eval_subset

        max_houses_per_split = {"train": 0, "val": 0, "test": 0}

        max_houses_per_split[subset_to_load] = int(1e9)
        return prior.load_dataset(
            dataset="spoc-data",
            entity="spoc-robot",
            revision="local-objaverse-procthor-houses",
            path_to_splits=None,
            split_to_path={
                k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz")
                for k in ["train", "val", "test"]
            },
            max_houses_per_split=max_houses_per_split,
        )[subset_to_load]

    def log_benchmark_stat(self):
        for task_type, samples in self.eval_samples.items():
            assert (
                map_hard_easy_objectnavtype_to_objectnavtype(task_type) in REGISTERED_TASKS
            ), f"Task type {task_type} not registered"
            self.log_data_stat(task_type=task_type, samples=self.eval_samples[task_type])

    def evaluate(self, agent_class, agent_input):
        self.workers = {}
        worker_to_gpu = {}

        self.log_benchmark_stat()

        # Initializing the workers
        for worker_id in range(self.num_workers):
            gpu_device = self.gpu_devices[worker_id % len(self.gpu_devices)]
            worker_to_gpu[worker_id] = gpu_device

            worker_args = {
                "gpu_device": gpu_device,
                "houses": self.houses,
                "max_eps_len": self.max_eps_len,
                "input_sensors": self.input_sensors,
                "skip_done": self.skip_done,
                "logging_sensor": self.logging_sensor,
                "outdir": self.outdir,
                "worker_id": worker_id,
                "det_type": self.det_type,
                "prob_randomize_lighting": self.prob_randomize_lighting,
                "prob_randomize_materials": self.prob_randomize_materials,
                "prob_randomize_colors": self.prob_randomize_colors,
            }

            self.workers[worker_id] = self.WorkerType(
                **worker_args,
            )

        max_len = max([len(vs) for vs in self.eval_samples.values()])
        inds = list(range(max_len))

        # Put all tasks and samples in the queue
        tasks_queue = mp.Queue()
        tasks_list = []
        timestamp = int(datetime.now().timestamp())
        dt = datetime.fromtimestamp(timestamp)
        timestamp = dt.strftime("%Y-%m-%d-%H:%M:%S")
        self.num_tasks_in_queue = 0
        num_tasks_per_type = {}
        for task_type, samples in self.eval_samples.items():
            num_tasks_per_type[task_type] = len(self.eval_samples[task_type])
            cur_random_seq = inds[: len(samples)]
            random.shuffle(cur_random_seq)

            for ind in cur_random_seq[: self.table_size]:
                samples[ind]["needs_video"] = True

            for sample in samples:
                tasks_queue.put(sample)
                tasks_list.append(sample)   
                self.num_tasks_in_queue += 1

        print(f"{self.num_tasks_in_queue} tasks in queue")

        results_queue = mp.Queue()
        procs = []

        # Starting the workers
        # the option of single worker and avoid distributed if num_workers == 1
        if len(self.workers) == 1:
            print(f"Starting worker in main process")

            worker_to_gpu = {0: self.gpu_devices[0]}
            start_worker(
                self.workers[0],
                agent_class,
                agent_input,
                device=worker_to_gpu[0],
                tasks_queue=tasks_queue,
                results_queue=results_queue,
                tasks_list=tasks_list,
            )
        else:
            for worker_id in self.workers:
                print(f"Starting worker {worker_id}")

                proc = mp.Process(
                    target=start_worker,
                    args=(
                        self.workers[worker_id],
                        agent_class,
                        agent_input,
                        worker_to_gpu[worker_id],
                        tasks_queue,
                        results_queue,
                        tasks_list,
                        timestamp,
                    ),
                )

                procs.append(proc)
                proc.start()

        self.join_processes_and_log(procs, results_queue, num_tasks_per_type)

    def accumulate_results(self, results_queue, task_type_to_results_list):
        new_found_results = 0
        while True:
            try:
                task_metrics_video_info = results_queue.get(timeout=10)
            except EmptyQueueError:
                # No more results to read for now - exit
                break

            if task_metrics_video_info is None:
                self.num_ended_workers += 1
                continue

            metrics_info, video_info = task_metrics_video_info
            task_type = metrics_info["task_type"]

            new_found_results += 1
            task_type_to_results_list[task_type].append(task_metrics_video_info)


        return new_found_results

    def join_processes_and_log(self, procs, results_queue, num_tasks_per_type):
        print("Logging and waiting for proccesses to finish")

        pd.set_option("display.width", 0)
        pd.set_option("max_colwidth", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.expand_frame_repr", False)

        total_tasks = sum(num_tasks_per_type.values())
        total_finished = 0
        start_time = None
        uncounted_finished = 0

        task_type_to_results_list = defaultdict(list)

        while True:
            if platform.system() != "Darwin":
                time.sleep(30)  # Wait 30 seconds before moving on to the next one

            new_found_results = self.accumulate_results(results_queue, task_type_to_results_list)

            current_time = time.time()
            total_finished += new_found_results

            if start_time is None:
                start_time = current_time
                uncounted_finished = new_found_results
            else:
                speed = (total_finished - uncounted_finished) / (current_time - start_time)
                ETA = (total_tasks - total_finished) / (speed + 1e-8)
                print(f"Rate {speed:.3f} tasks/s, ETA {ETA / 3600:.3f}hours")

            num_alive = 0
            dead_proc_inds = []
            for index, proc in enumerate(procs):
                if proc is None:
                    continue

                if proc.is_alive():
                    num_alive += 1
                    continue

                # Process has finished
                dead_proc_inds.append(index)
                try:
                    with time_limit(30):
                        proc.join()  # Wait for the process to finish
                        print(f"Joined worker index {index}")
                except TimeoutException:
                    print("Timed out!", index)
                finally:
                    procs[index] = None

            if num_alive == 0 or self.num_ended_workers == len(procs):
                print(f"No process alive. Terminating")
                break
            elif num_alive == 1 and len(dead_proc_inds) == 0 and results_queue.qsize() == 0:
                print(f"1 process alive but no results in queue. Terminating")
                break
            else:
                print(f"{num_alive} alive eval workers")

            if (new_found_results > 0 or len(dead_proc_inds) > 0) and LOG_INTERMEDIATE_RESULTS:
                self.log_from_task_type_lists(
                    task_type_to_results_list,
                    upload=True,
                    upload_video=len(dead_proc_inds) > 0,
                    upload_per_task=len(dead_proc_inds) > 0,
                    upload_per_synset=True,
                )

        print("Evaluation finished")

        # Gathering missing results
        _ = self.accumulate_results(results_queue, task_type_to_results_list)

        # Log whatever we found
        self.log_from_task_type_lists(task_type_to_results_list, upload=True)
    def log_from_task_type_lists(
        self,
        task_type_to_results_list,
        upload=True,
        upload_per_task=None,
        upload_video=None,
        upload_per_synset=None,
    ):
        all_tasks_aggregated_results = {}
        table_plan2 = self.wandb.Table(columns=["sample_id", "cost","corner", "danger", "blind", "fragile", "critical", "success"])
        for task_type, task_type_results in task_type_to_results_list.items():
                for result in task_type_results:
                    table_plan2.add_data(result[0]["sample_id"],result[0]['metrics']["cost"], result[0]['metrics']["corner"], result[0]['metrics']["danger"], \
                        result[0]['metrics']["blind"], result[0]['metrics']["fragile"], result[0]['metrics']["critical"], int(result[0]['metrics']["success"]))
        # self.wandb.log({"plan1": table_plan1})
        self.wandb.log({f"plan2": table_plan2})

        for task_type, task_type_results in task_type_to_results_list.items():
            all_tasks_aggregated_results[task_type] = log_results(
                wandb=self.wandb,
                all_workers_results=task_type_results,
                task_type=task_type,
                upload_per_task=upload_per_task if upload_per_task is not None else upload,
                upload_video=upload_video if upload_video is not None else upload,
                upload_per_synset=upload_per_synset if upload_per_synset is not None else upload,
            )

        self.log_aggregated_results(all_tasks_aggregated_results, upload=upload)

    def log_aggregated_results(self, all_tasks_aggregated_results, upload=True):
        if all_tasks_aggregated_results == {}:
            return
        metrics_to_log, _ = all_tasks_aggregated_results[
            list(all_tasks_aggregated_results.keys())[0]
        ]
        metrics_to_log = [
            m for m in metrics_to_log if "extra/" not in m and "for_video_table/" not in m
        ]
        columns = ["task_type"] + ["extra_tag"] + metrics_to_log
        aggrgeated_result_metrics_table = self.wandb.Table(columns=columns)
        total_results = 0
        for task_type, results in all_tasks_aggregated_results.items():
            total_results += len(results)
            metrics_to_log, metric_values = results
            metric_dict = {metrics_to_log[i]: metric_values[i] for i in range(len(metrics_to_log))}
            this_row = []
            for col in columns:
                if col == "task_type":
                    this_row.append(task_type)
                elif col == "extra_tag":
                    this_row.append(self.extra_tag)
                elif col in metric_dict:
                    this_row.append(metric_dict[col])
                else:
                    this_row.append(-1)
                    print("missing metric", col, "for task", task_type)

            aggrgeated_result_metrics_table.add_data(*this_row)

        print("\nAggregated results")
        print(aggrgeated_result_metrics_table.get_dataframe())

        if upload:
            self.wandb.log({f"FullAggregatedResults": aggrgeated_result_metrics_table})
            print(
                f"Uploading results from {total_results} tasks out of {self.num_tasks_in_queue} emitted."
            )
