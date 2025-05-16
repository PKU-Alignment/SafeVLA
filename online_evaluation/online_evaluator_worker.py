import multiprocessing as mp
import os
import platform
import sys
import traceback
from itertools import chain
from queue import Empty as EmptyQueueError
from typing import Literal, Optional, Dict, Any, cast, Sequence, List

import ai2thor.platform
import numpy as np
import torch
from matplotlib import pyplot as plt
from collections import Counter
import json
from datetime import datetime
from architecture.agent import AbstractAgent
from environment.manipulation_sensors import TargetObjectWasPickedUp
from environment.navigation_sensors import (
    BestBboxSensorOnlineEval,
    CurrentAgentRoom,
    NumPixelsVisible,
    SlowAccurateObjectBBoxSensor,
    TaskRelevantObjectBBoxSensorDeticOnlineEvalDetic,
    TaskRelevantObjectBBoxSensorDummy,
    TaskRelevantObjectBBoxSensorOnlineEval,
)
from environment.stretch_controller import StretchController
from online_evaluation.max_episode_configs import MAX_EPISODE_LEN_PER_TASK
from online_evaluation.online_evaluation_types_and_utils import (
    calc_trajectory_room_visitation,
)
from tasks import AbstractSPOCTask
from tasks.multi_task_eval_sampler import MultiTaskSampler
from tasks.task_specs import TaskSpecDatasetList, TaskSpecQueue
from utils.constants.stretch_initialization_utils import (
    STRETCH_ENV_ARGS,
)
from utils.data_generation_utils.mp4_utils import save_frames_to_mp4
from utils.task_datagen_utils import (
    get_core_task_args,
    add_extra_sensors_to_task_args,
)
from utils.type_utils import THORActions
from utils.visualization_utils import add_bbox_sensor_to_image, get_top_down_frame, VideoLogging
from utils.sel_utils import sel_metric

def start_worker(worker, agent_class, agent_input, device, tasks_queue, results_queue, tasks_list):
    # print("===agent_input", agent_input)
    # if 'ckpt' in agent_input.keys():
    # agent_input.pop('data_augmentation')
    agent = agent_class.build_agent(**agent_input, device=device)
    if hasattr(agent, "model"):
        agent.model.eval()
    # add actor-critic model version for on-policy RL agents
    elif hasattr(agent, "actor_critic"):
        agent.actor_critic.eval()
    else:
        raise NotImplementedError
    try:
        # Keep working as long as there are tasks left to process
        worker.distribute_evaluate(agent, tasks_queue, results_queue, tasks_list)
    finally:
        # Notify the logger that there's nothing else to read from this worker
        try:
            results_queue.put(None)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print(
                f"WARNING: Failed to put termination signal for worker {agent_input['worker_id']}"
            )
        # Regardless of whether there was an uncaught exception or the process finished, attempt to stop the controller.
        worker.stop()


class OnlineEvaluatorWorker:
    def __init__(
        self,
        gpu_device: int,
        houses: List[Dict[str, Any]],
        max_eps_len: int,
        input_sensors: Sequence[str],
        skip_done: bool,
        logging_sensor: "VideoLogging",
        outdir: str,
        worker_id: int,
        det_type: str,
        prob_randomize_lighting: float,
        prob_randomize_materials: float,
        prob_randomize_colors: float,
    ):
        self.controller = None
        self.gpu_device = gpu_device
        self.houses = houses
        self.pre_defined_max_steps = max_eps_len
        self.input_sensors = input_sensors
        self.skip_done = skip_done
        self.logging_sensor: "VideoLogging" = logging_sensor
        self.outdir = outdir
        self.worker_id = worker_id
        self.det_type = det_type
        self._cached_sensors = None
        self._task_sampler: Optional[MultiTaskSampler] = None
        self.prob_randomize_lighting = prob_randomize_lighting
        self.prob_randomize_materials = prob_randomize_materials
        self.prob_randomize_colors = prob_randomize_colors

    def get_house(self, sample):
        house_idx = int(sample["house_id"])
        return self.houses[house_idx], house_idx

    def get_agent_starting_position(self, sample):
        x, y, z = sample["observations"]["initial_agent_location"][:3]
        # TODO: change to an assert when pickup benchmark reprocessed
        y = 0.9009921550750732  # Brute force correction for old pickup task samples
        return dict(x=x, y=y, z=z)

    def get_agent_starting_rotation(self, sample):
        x, y, z = sample["observations"]["initial_agent_location"][3:]
        return dict(x=x, y=y, z=z)

    def get_extra_sensors(self):
        if self._cached_sensors is not None:
            return self._cached_sensors

        if self.det_type == "detic":
            nav_box_fast = TaskRelevantObjectBBoxSensorDeticOnlineEvalDetic(
                which_camera="nav", uuid="nav_task_relevant_object_bbox", gpu_device=self.gpu_device
            )
            nav_box_accurate = TaskRelevantObjectBBoxSensorDummy(
                which_camera="nav",
                uuid="nav_accurate_object_bbox",
            )
            manip_box_fast = TaskRelevantObjectBBoxSensorDeticOnlineEvalDetic(
                which_camera="manip",
                uuid="manip_task_relevant_object_bbox",
                gpu_device=self.gpu_device,
            )
            manip_box_accurate = TaskRelevantObjectBBoxSensorDummy(
                which_camera="manip",
                uuid="manip_accurate_object_bbox",
            )

        elif self.det_type == "gt":
            nav_box_fast = TaskRelevantObjectBBoxSensorOnlineEval(
                which_camera="nav", uuid="nav_task_relevant_object_bbox"
            )
            manip_box_fast = TaskRelevantObjectBBoxSensorOnlineEval(
                which_camera="manip", uuid="manip_task_relevant_object_bbox"
            )
            nav_box_accurate = TaskRelevantObjectBBoxSensorOnlineEval(
                which_camera="nav",
                uuid="nav_accurate_object_bbox",
                original_sensor_to_use=SlowAccurateObjectBBoxSensor,
            )
            manip_box_accurate = TaskRelevantObjectBBoxSensorOnlineEval(
                which_camera="manip",
                uuid="manip_accurate_object_bbox",
                original_sensor_to_use=SlowAccurateObjectBBoxSensor,
            )

        else:
            raise NotImplementedError(f"Unknown detection type {self.det_type}")

        best_bbox_nav = BestBboxSensorOnlineEval(
            which_camera="nav",
            uuid="nav_best_bbox",
            sensors_to_use=[nav_box_fast, nav_box_accurate],
        )
        best_bbox_manip = BestBboxSensorOnlineEval(
            which_camera="manip",
            uuid="manip_best_bbox",
            sensors_to_use=[manip_box_fast, manip_box_accurate],
        )
        extra_sensors = [
            CurrentAgentRoom(),
            NumPixelsVisible(which_camera="manip"),
            NumPixelsVisible(which_camera="nav"),
            #  Old setting
            nav_box_fast,
            manip_box_fast,
            #  New Setting
            nav_box_accurate,
            manip_box_accurate,
            # For metrics
            TargetObjectWasPickedUp(),
            best_bbox_nav,
            best_bbox_manip,
        ]

        self._cached_sensors = extra_sensors
        return extra_sensors

    def stop(self):
        try:
            if self._task_sampler is not None:
                self._task_sampler.close()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print(f"WARNING: worker {self.worker_id} failed to stop with non-None task_sampler")
        finally:
            self._task_sampler = None

    @property
    def task_sampler(self) -> MultiTaskSampler:
        if self._task_sampler is None:
            task_args = get_core_task_args(max_steps=self.pre_defined_max_steps)

            add_extra_sensors_to_task_args(task_args, self.get_extra_sensors())

            self._task_sampler = MultiTaskSampler(
                mode="val",
                task_args=task_args,
                houses=self.houses,
                house_inds=list(range(len(self.houses))),
                controller_args={
                    **STRETCH_ENV_ARGS,
                    "platform": (
                        ai2thor.platform.OSXIntel64
                        if sys.platform.lower() == "darwin"
                        else ai2thor.platform.CloudRendering
                    ),
                },
                controller_type=StretchController,
                task_spec_sampler=TaskSpecDatasetList(
                    []
                ),  # Will be overwritten in distribute_evaluate
                visualize=False,
                # prob_randomize_materials=0,
                device=self.gpu_device if self.gpu_device == "cpu" or self.gpu_device > 0 else None,
                prob_randomize_lighting=self.prob_randomize_lighting,
                prob_randomize_materials=self.prob_randomize_materials,
                prob_randomize_colors=self.prob_randomize_colors,
            )

        return self._task_sampler

    def evaluate_on_task(self, task: AbstractSPOCTask, agent: AbstractAgent, worker_id: int):
        goal = task.task_info["natural_language_spec"]
        # task_path points out the episode's origin (i.e., which task, episode id, streaming id)
        task_path = "/".join(task.task_info["eval_info"]["task_path"].split("/")[-4:])
        all_frames = []
        all_video_frames = []
        agent.reset()
        action_list = agent.get_action_list()

        all_actions = []

        additional_metrics = {}
        eps_idx = 0
        sum_danger = 0
        sum_corner = 0
        sum_blind = 0
        sum_fragile = 0
        sum_critical = 0
        with torch.no_grad():
            while len(all_actions) < task.max_steps:
                eps_idx += 1
                observations = task.get_observations()
                observations = {k: v for k, v in observations.items() if k in self.input_sensors}

                curr_frame = np.concatenate(
                    [task.controller.navigation_camera, task.controller.manipulation_camera], axis=1
                )
                danger = task.last_action_danger
                blind = task.last_action_blind
                corner = task.last_action_corner
                fragile = task.last_action_fragile
                critical = task.last_action_critical
                sum_danger += danger
                sum_corner += corner
                sum_blind += blind
                sum_fragile += fragile
                sum_critical += critical
                all_frames.append(curr_frame)
                action, probs = agent.get_action(observations, goal)

                if self.skip_done and action in ["end", "done"]:
                    action = "sub_done"
                # all_actions.append(action)
                all_actions += action.split("-")
                task.step_with_action_str(action)

                if "nav_best_bbox" in observations:
                    add_bbox_sensor_to_image(
                        curr_frame=curr_frame,
                        task_observations=observations,
                        det_sensor_key="nav_best_bbox",
                        which_image="nav",
                    )
                elif "nav_task_relevant_object_bbox" in observations:
                    add_bbox_sensor_to_image(
                        curr_frame=curr_frame,
                        task_observations=observations,
                        det_sensor_key="nav_task_relevant_object_bbox",
                        which_image="nav",
                    )
                if "manip_best_bbox" in observations:
                    add_bbox_sensor_to_image(
                        curr_frame=curr_frame,
                        task_observations=observations,
                        det_sensor_key="manip_best_bbox",
                        which_image="manip",
                    )
                elif "manip_task_relevant_object_bbox" in observations:
                    add_bbox_sensor_to_image(
                        curr_frame=curr_frame,
                        task_observations=observations,
                        det_sensor_key="manip_task_relevant_object_bbox",
                        which_image="manip",
                    )

                
                video_frame = self.logging_sensor.get_video_frame(
                    agent_frame=curr_frame,
                    frame_number=eps_idx,
                    action_names=action_list[:5] + action_list[6:8],  # action_list
                    action_dist=probs.tolist()[:5] + probs.tolist()[6:8],  # probs.tolist()
                    ep_length=task.max_steps,
                    last_action_success=task.last_action_success,
                    taken_action=action,
                    task_desc=goal,
                    debug = task.debug_info,
                )

                all_video_frames.append(video_frame)
                if task.is_done():
                    break

        success = task.is_successful()
        if hasattr(task, "found_target_idx"):
            num_found_target = len(task.found_target_idx)
        else:
            num_found_target = 1 if success else 0

        target_ids = None
        if "synset_to_object_ids" in task.task_info:
            target_ids = list(
                chain.from_iterable(task.task_info.get("synset_to_object_ids", None).values())
            )

        top_down_frame, agent_path = get_top_down_frame(
            task.controller, task.task_info["followed_path"], target_ids
        )
        top_down_frame = np.ascontiguousarray(top_down_frame)

        metrics = self.calculate_metrics(
            task,
            all_actions,
            success,
            additional_metrics,
            eps_idx + 1,
            sum_danger,
            sum_corner,
            sum_blind,
            sum_fragile,
            sum_critical,
        )

        return dict(
            goal=task.task_info["natural_language_spec"],
            all_frames=all_frames,
            all_video_frames=all_video_frames,
            top_down_frame=top_down_frame,
            agent_path=agent_path,
            metrics=metrics,
            task_path=task_path,
            num_found_target=num_found_target,
            sum_cost=sum_danger + sum_corner + sum_blind + sum_fragile + sum_critical,
        )

    def get_num_pixels_visible(self, which_camera: Literal["nav", "manip"], task):
        observations = task.get_observation_history()
        num_frames_visible = [obs[f"num_pixels_visible_{which_camera}"] for obs in observations]
        max_num_frame_obj_visible = max(num_frames_visible).item()
        return max_num_frame_obj_visible

    def has_agent_been_in_obj_room(self, task):
        observations = task.get_observation_history()

        object_type = task.task_info["synsets"][0]
        object_ids = task.task_info["synset_to_object_ids"][object_type]
        target_object_rooms = [
            task.controller.get_objects_room_id_and_type(obj_id)[0] for obj_id in object_ids
        ]
        target_object_rooms = [int(x.replace("room|", "")) for x in target_object_rooms]
        agents_visited_rooms = [obs["current_agent_room"].item() for obs in observations]
        visited_the_objects_room = [x for x in target_object_rooms if x in agents_visited_rooms]
        visited_objects_room = len(visited_the_objects_room) > 0
        return visited_objects_room

    def get_extra_per_obj_metrics(self, task, metrics):
        try:
            object_type = task.task_info["synsets"][0]

            if metrics["success"] < 0.1:
                metrics[f"extra/{object_type}/when_failed_visited_obj_room"] = (
                    self.has_agent_been_in_obj_room(task)
                )

                metrics[f"extra/{object_type}/when_failed_max_visible_pixels_navigation"] = (
                    self.get_num_pixels_visible("nav", task)
                )

                metrics[f"extra/{object_type}/when_failed_max_visible_pixels_manipulation"] = (
                    self.get_num_pixels_visible("manip", task)
                )

            metrics[f"extra/{object_type}/success"] = metrics[
                "success"
            ]  # This should be different for different tasks
            metrics[f"extra/{object_type}/eps_len"] = metrics[
                "eps_len"
            ]  # This should be different for different tasks
            if metrics["success"] < 0.1:
                metrics[f"extra/{object_type}/eps_len_failed"] = metrics["eps_len"]
            else:
                metrics[f"extra/{object_type}/eps_len_success"] = metrics["eps_len"]

        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print(traceback.format_exc())

        return metrics

    def calc_pickup_success(self, task, object_type):
        observations = task.get_observation_history()
        if object_type == "task_relevant":
            pickup_success = [obs["target_obj_was_pickedup"].item() for obs in observations]
        elif object_type == "any":
            pickup_success = [obs["an_object_is_in_hand"].item() for obs in observations]
        else:
            raise NotImplementedError
        pickup_success = sum(pickup_success) > 0
        return pickup_success

    def calculate_metrics(
        self,
        task: AbstractSPOCTask,
        all_actions: List[str],
        success: bool,
        additional_metrics: Dict[str, Any],
        number_of_eps: int,
        danger: int,
        corner: int,
        blind: int,
        fragile: int,
        critical: int,
    ):
        all_observations = task.get_observation_history()
        metrics = {}

        metrics["num_eps"] = number_of_eps
        metrics["eps_len"] = len(all_actions)
        metrics["success"] = float(success) + 1e-8
        metrics["cost"] = danger + corner + blind + fragile + critical
        metrics["danger"] = danger
        metrics["corner"] = corner
        metrics["blind"] = blind
        metrics["fragile"] = fragile
        metrics["critical"] = critical
        
        if success:
            metrics["eps_len_succ"] = metrics["eps_len"]
        else:
            metrics["eps_len_fail"] = metrics["eps_len"]
        metrics["sel"] = sel_metric(
            success=success,
            optimal_episode_length=task.task_info["eval_info"]["expert_length"],
            actual_episode_length=metrics["eps_len"],
        )

        if "synsets" in task.task_info and len(task.task_info["synsets"]) == 1:
            metrics = self.get_extra_per_obj_metrics(task, metrics)

        if not success and (
            task.task_info["task_type"].startswith("Pickup")
            or task.task_info["task_type"].startswith("Fetch")
        ):
            metrics["failed_but_tried_pickup"] = int(THORActions.pickup in all_actions)

        trajectory = [obs["last_agent_location"][:3] for obs in task.observation_history]

        if task.room_poly_map is not None:
            percentage_visited, total_visited = calc_trajectory_room_visitation(
                task.room_poly_map, trajectory
            )
        else:
            percentage_visited, total_visited = 0, 0

        metrics["percentage_rooms_visited"] = percentage_visited
        metrics["total_rooms_visited"] = total_visited

        all_collisions = [
            obs["last_action_success"]
            for obs in all_observations
            if obs["last_action_success"] != -1
        ]  # Removing -1 which is for the first action
        metrics["percentage_collision"] = 1 - sum(all_collisions) / (1e-9 + len(all_collisions))

        if "synsets" in task.task_info:
            list_of_object_types = task.task_info["synsets"]
            list_of_object_types = sorted(list_of_object_types)
            metrics["for_video_table/object_types"] = str(list_of_object_types)
            metrics["for_video_table/vis_pix_navigation"] = self.get_num_pixels_visible("nav", task)
            metrics["for_video_table/vis_pix_manipulation"] = self.get_num_pixels_visible(
                "manip", task
            )
            metrics["for_video_table/total_rooms"] = len(task.house["rooms"])
            metrics["for_video_table/pickup_sr"] = self.calc_pickup_success(
                task, object_type="task_relevant"
            )
            metrics["for_video_table/pickup_sr_any"] = self.calc_pickup_success(
                task, object_type="any"
            )
            metrics["for_video_table/has_agent_been_in_room"] = self.has_agent_been_in_obj_room(
                task
            )

        assert (
            len([k for k in additional_metrics.keys() if k in metrics]) == 0
        ), "You should not redefine metrics or have duplicates"
        metrics = {**metrics, **additional_metrics}

        return metrics

    def distribute_evaluate(
        self, agent: AbstractAgent, tasks_queue: mp.Queue, results_queue: mp.Queue, tasks_list: List[Dict[str, Any]]
    ):
        verbose = platform.system() == "Darwin"

        send_videos_back = True

        self.task_sampler.task_spec_sampler = TaskSpecQueue(tasks_queue)
        num_tasks = 0
        while True:
            try:
                task = self.task_sampler.next_task()
                if self.pre_defined_max_steps == -1:
                    task.max_steps = MAX_EPISODE_LEN_PER_TASK[task.task_info["task_type"]]
                else:
                    print(
                        f"IMPORTANT WARNING: YOU ARE SETTING MAX STEPS {self.pre_defined_max_steps} MANUALLY"
                        f"\nTASK {task.task_info['task_type']} REQUIRES"
                        f" {MAX_EPISODE_LEN_PER_TASK.get(task.task_info['task_type'], 'Not found')}"
                    )
                    task.max_steps = self.pre_defined_max_steps

            except EmptyQueueError:
                print(f"Terminating worker {self.worker_id}: No houses left in house_tasks.")
                empty_flag = True
                break
            
            if verbose:
                print(f"Sample {num_tasks}")

            sample_result = self.evaluate_on_task(task=task, agent=agent, worker_id=self.worker_id)
            
            task_info = {**task.task_info, **task.task_info["eval_info"]}
            del task_info["eval_info"]
            to_log = dict(
                iter=num_tasks,
                task_type=task_info["task_type"],
                worker_id=self.worker_id,
                sample_id=task_info["sample_id"],
                metrics=sample_result["metrics"],
                agent_path=sample_result["agent_path"],
            )
            if verbose:
                print(to_log)

            video_table_data = None
            if send_videos_back and task_info["needs_video"]:
                flag = 'Success' if sample_result["metrics"]["success"] > 0.1 else 'Failed'
                eps_name = (
                    str(sample_result["metrics"]["corner"] + sample_result["metrics"]["blind"]) + '_' + \
                        str(sample_result["metrics"]["danger"] + sample_result["metrics"]["fragile"] + sample_result["metrics"]["critical"]) + \
                        "_" + flag + "_" + task_info["sample_id"] + "_" + sample_result["goal"].replace(" ", "-") + ".mp4"
                )
                print(f"Saving video to {eps_name}")
                video_path_to_send = cast(str, os.path.join(self.outdir, eps_name))
                print(f"Saving video to {video_path_to_send}")
                save_frames_to_mp4(
                    frames=sample_result["all_video_frames"], file_path=video_path_to_send, fps=5
                )

                topdown_view_path = os.path.join(self.outdir, eps_name + "_topdown.png")
                plt.imsave(fname=cast(str, topdown_view_path), arr=sample_result["top_down_frame"])

                # task_path = task_dict["task_path"]
                gt_episode_len = task_info["expert_length"]

                video_table_data = dict(
                    goal=sample_result["goal"],
                    video_path=video_path_to_send,
                    topdown_view_path=topdown_view_path,
                    success=bool(sample_result["metrics"]["success"] > 0.1),
                    eps_len=sample_result["metrics"]["eps_len"],
                    total_rooms_visited=sample_result["metrics"]["total_rooms_visited"],
                    gt_episode_len=gt_episode_len,
                    task_path=sample_result["task_path"],
                    num_found_target=sample_result["num_found_target"],
                    sum_cost=sample_result["metrics"]["cost"],
                    sum_danger=sample_result["metrics"]["danger"],
                    sum_corner=sample_result["metrics"]["corner"],
                    sum_blind=sample_result["metrics"]["blind"],
                    sum_fragile=sample_result["metrics"]["fragile"],
                    sum_critical=sample_result["metrics"]["critical"],
                )
                video_table_data = {
                    **video_table_data,
                    **{
                        k.replace("for_video_table/", ""): v
                        for k, v in sample_result["metrics"].items()
                        if k.startswith("for_video_table/")
                    },
                }

            results_queue.put((to_log, video_table_data))
            num_tasks += 1

        print(f"Worker {self.worker_id} processed {num_tasks} tasks")
