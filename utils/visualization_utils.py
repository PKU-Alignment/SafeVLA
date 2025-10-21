import copy
from typing import Sequence, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import ImageFont, Image, ImageDraw

from environment.stretch_controller import StretchController
from utils.constants.stretch_initialization_utils import stretch_long_names

DISTINCT_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 0, 0),  # Dark Red
    (0, 128, 0),  # Dark Green
    (0, 0, 128),  # Dark Blue
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (255, 165, 0),  # Orange
    (255, 192, 203),  # Pink
    (255, 255, 255),  # White
    (0, 0, 0),  # Black
    (0, 0, 139),  # DarkBlue
    (0, 100, 0),  # DarkGreen
    (139, 0, 139),  # DarkMagenta
    (165, 42, 42),  # Brown
    (255, 215, 0),  # Gold
    (64, 224, 208),  # Turquoise
    (240, 230, 140),  # Khaki
    (70, 130, 180),  # Steel Blue
]


def add_bboxes_to_frame(
    frame: np.ndarray,
    bboxes: Sequence[Sequence[float]],
    labels: Optional[Sequence[str]],
    inplace=False,
    colors=tuple(DISTINCT_COLORS),
    thinkness=1,
):
    """
    Visualize bounding boxes on an image and save the image to disk.

    Parameters:
    - frame: numpy array of shape (height, width, 3) representing the image.
    - bboxes: list of bounding boxes. Each bounding box is a list of [min_row, min_col, max_row, max_col].
    - labels: list of labels corresponding to each bounding box.
    - inplace: whether to modify the input frame in place or return a new frame.
    """
    # Convert numpy image to PIL Image for visualization

    assert frame.dtype == np.uint8
    if not inplace:
        frame = copy.deepcopy(frame)

    bboxes_cleaned = [[int(v) for v in bbox] for bbox in bboxes if -1 not in bbox]
    if labels is None:
        labels = [""] * len(bboxes_cleaned)

    h, w, _ = frame.shape

    # Plot bounding boxes and labels
    for bbox, label, color in zip(bboxes_cleaned, labels, colors):
        if np.all(bbox == 0):
            continue
        cv2.rectangle(frame, bbox[:2], bbox[2:], color=color, thickness=thinkness)

        cv2.putText(
            frame,
            label,
            (int(bbox[0]), int(bbox[1] + 15)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=2,
        )

    return frame


def add_bbox_sequence_to_frame_sequence(frames, double_bboxes):
    T, num_coords = double_bboxes.shape
    assert num_coords == 10
    assert T == len(frames)

    convert_to_torch = False
    if torch.is_tensor(frames):
        frames = frames.numpy()
        convert_to_torch = True

    double_bboxes[double_bboxes == 1000] = 0

    for i, frame in enumerate(frames):
        bbox_list = [double_bboxes[i][:4], double_bboxes[i][5:9]]
        add_bboxes_to_frame(
            frame,
            bbox_list,
            labels=None,
            inplace=True,
            colors=[(255, 0, 0), (0, 255, 0)],
            thinkness=2,
        )
    if convert_to_torch:
        result = torch.Tensor(frames).to(torch.uint8)
    else:
        result = frames
    return result


def add_bbox_sensor_to_image(curr_frame, task_observations, det_sensor_key, which_image):
    task_relevant_object_bbox = task_observations[det_sensor_key]
    (bbox_dim,) = task_relevant_object_bbox.shape
    assert bbox_dim in [5, 10]
    if bbox_dim == 5:
        task_relevant_object_bboxes = [task_relevant_object_bbox[:4]]
    if bbox_dim == 10:
        task_relevant_object_bboxes = [
            task_relevant_object_bbox[:4],
            task_relevant_object_bbox[5:9],
        ]
        task_relevant_object_bboxes = [
            b for b in task_relevant_object_bboxes if b[1] <= curr_frame.shape[0]
        ]
    if which_image == "nav":
        pass
    elif which_image == "manip":
        start_index = curr_frame.shape[1] // 2
        for i in range(len(task_relevant_object_bboxes)):
            task_relevant_object_bboxes[i][0] += start_index
            task_relevant_object_bboxes[i][2] += start_index
    else:
        raise NotImplementedError
    if len(task_relevant_object_bboxes) > 0:
        # This works because the navigation frame comes first in curr_frame
        add_bboxes_to_frame(
            frame=curr_frame,
            bboxes=task_relevant_object_bboxes,
            labels=None,
            inplace=True,
        )


def get_top_down_path_view(
    controller: StretchController,
    agent_path: Sequence[Dict[str, float]],
    targets_to_highlight=None,
    orthographic: bool = True,
    map_height_width=(1000, 1000),
    path_width: float = 0.045,
):
    thor_controller = controller.controller

    original_hw = thor_controller.last_event.frame.shape[:2]

    if original_hw != map_height_width:
        event = thor_controller.step(
            "ChangeResolution", x=map_height_width[1], y=map_height_width[0], raise_for_failure=True
        )

    if len(thor_controller.last_event.third_party_camera_frames) < 2:
        event = thor_controller.step("GetMapViewCameraProperties", raise_for_failure=True)
        cam = copy.deepcopy(event.metadata["actionReturn"])
        if not orthographic:
            bounds = event.metadata["sceneBounds"]["size"]
            max_bound = max(bounds["x"], bounds["z"])

            cam["fieldOfView"] = 50
            cam["position"]["y"] += 1.1 * max_bound
            cam["orthographic"] = False
            cam["farClippingPlane"] = 50
            del cam["orthographicSize"]

        event = thor_controller.step(
            action="AddThirdPartyCamera",
            **cam,
            skyboxColor="white",
            raise_for_failure=True,
        )

    waypoints = []
    for target in targets_to_highlight or []:
        target_position = controller.get_object_position(target)
        target_dict = {
            "position": target_position,
            "color": {"r": 1, "g": 0, "b": 0, "a": 1},
            "radius": 0.5,
            "text": "",
        }
        waypoints.append(target_dict)

    if len(agent_path) != 0:
        thor_controller.step(
            action="VisualizeWaypoints",
            waypoints=waypoints,
            raise_for_failure=True,
        )
        # put this over the waypoints just in case
        event = thor_controller.step(
            action="VisualizePath",
            positions=agent_path,
            pathWidth=path_width,
            raise_for_failure=True,
        )
        thor_controller.step({"action": "HideVisualizedPath"})

    map = event.third_party_camera_frames[-1]

    if original_hw != map_height_width:
        thor_controller.step(
            "ChangeResolution", x=original_hw[1], y=original_hw[0], raise_for_failure=True
        )

    return map

# TODO: plan1 save the path with unsafe points
def get_top_down_frame(controller, agent_path, target_ids):
    top_down, agent_path = controller.get_top_down_path_view(agent_path, target_ids)
    return top_down, agent_path


class VideoLogging:
    @staticmethod
    def get_video_frame(
        agent_frame: np.ndarray,
        frame_number: int,
        action_names: List[str],
        action_dist: Optional[List[float]],
        ep_length: int,
        last_action_success: Optional[bool],
        taken_action: Optional[str],
        task_desc: str,
        debug : Optional[any],
    ) -> np.array:
        agent_height, agent_width, ch = agent_frame.shape

        font_to_use = "Arial.ttf"  # possibly need a full path here
        full_font_load = ImageFont.truetype(font_to_use, 14)

        IMAGE_BORDER = 25
        TEXT_OFFSET_H = 60  # Restore original value for action details
        TEXT_OFFSET_V = 30

        # Define two alignment positions - one for labels and one for action details
        action_x = IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H  # For action details
        info_x = IMAGE_BORDER * 2 + agent_width + 20  # For main labels alignment

        image_dims = (
            agent_height + 2 * IMAGE_BORDER + 30,
            agent_width + 2 * IMAGE_BORDER + 300,  # Further reduced right padding to 300
            ch,
        )
        image = np.full(image_dims, 255, dtype=np.uint8)
        image[
            IMAGE_BORDER : IMAGE_BORDER + agent_height, IMAGE_BORDER : IMAGE_BORDER + agent_width, :
        ] = agent_frame

        text_image = Image.fromarray(image)
        img_draw = ImageDraw.Draw(text_image)
        sum_cost = debug.get("sum_cost", None)
        sum_danger = debug.get("sum_danger", None)
        sum_corner = debug.get("sum_corner", None)
        sum_blind = debug.get("sum_blind", None)
        sum_critical = debug.get("sum_critical", None)
        sum_fragile = debug.get("sum_fragile", None)
        camera_seen = debug.get("camera_seen", None)    
        critical_objects = debug.get("critical_objects", None)  
        danger_objects = debug.get("danger_objects", None)  
        error_message = debug.get("error_message", None)  
        
        # Removed camera seen objects display for cleaner visualization
        if action_dist is not None:
            for i, (prob, action) in enumerate(zip(action_dist, action_names)):
                try:
                    action_long_name = stretch_long_names[action]
                except KeyError:
                    action_long_name = action
                if i < 10:
                    img_draw.text(
                    (
                        action_x,
                        (TEXT_OFFSET_V + 5) + i * 10,
                    ),
                        action_long_name,
                        font=ImageFont.truetype(font_to_use, 10),
                        fill="gray" if action != taken_action else "black",
                        anchor="rm",
                    )
                    img_draw.rectangle(
                        (
                            action_x + 5,
                            TEXT_OFFSET_V + i * 10,
                            IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 5) + int(100 * prob),
                            (TEXT_OFFSET_V + 5) + i * 10,
                        ),
                        outline="blue",
                        fill="blue",
                    )
                else:
                    img_draw.text(
                        (
                            action_x + 200,
                            (TEXT_OFFSET_V + 5) + (i - 10) * 10,
                        ),
                        action_long_name,
                        font=ImageFont.truetype(font_to_use, 10),
                        fill="gray" if action != taken_action else "black",
                        anchor="rm",
                    )
                    img_draw.rectangle(
                        (
                            action_x + 205,
                            TEXT_OFFSET_V + (i - 10) * 10,
                            IMAGE_BORDER * 2
                            + agent_width
                            + (TEXT_OFFSET_H + 205)
                            + int(100 * prob),
                            (TEXT_OFFSET_V + 5) + (i - 10) * 10,
                        ),
                        outline="blue",
                        fill="blue",
                    )

        img_draw.text(
            (info_x, IMAGE_BORDER * 1 + 90),
            f"Task: {task_desc}",
            font=full_font_load,
            fill=(100, 100, 100),  # Dark gray for consistency
            anchor="lm",
        )
        img_draw.text(
            (IMAGE_BORDER * 1.1, IMAGE_BORDER * 1),
            str(frame_number),
            font=full_font_load,  # ImageFont.truetype(font_to_use, 25),
            fill="white",
        )
        if last_action_success is not None:
            img_draw.text(
                (info_x, IMAGE_BORDER * 1 + 110),
                "Last Action:", 
                font=full_font_load,
                fill=(100, 100, 100),  # Dark gray for consistency
                anchor="lm",
            )
            img_draw.text(
                (info_x + 120, IMAGE_BORDER * 1 + 110),
                "Success" if last_action_success else "Failure",
                font=full_font_load,
                fill=(50, 180, 50) if last_action_success else (220, 50, 50),  # Adjusted colors
                anchor="lm",
            )

        if error_message is not None and error_message != '':
            split_char = '\''
            error_obj = None
            try:
                error_obj = error_message.split(split_char)[1]
            except:
                error_obj = error_message
            img_draw.text(
                (info_x, IMAGE_BORDER * 1 + 130),
                f"Error: {error_obj}",
                font=full_font_load,
                fill=(220, 50, 50),  # Consistent red color
                anchor="lm",
            )
        # Cost Information Display
        cost_start_y = IMAGE_BORDER * 1 + 175
        cost_spacing = 18  # Reduced from 22 to 18 to make table more compact
        title_spacing = 25  # Reduced from 30 to 25
        
        # Draw section title with line
        title_y = cost_start_y - title_spacing
        img_draw.text(
            (info_x, title_y),
            "Safety Metrics",
            font=ImageFont.truetype(font_to_use, 16),
            fill=(50, 50, 50),  # Dark gray for title
            anchor="lm",
        )
        # Draw horizontal line under title
        line_y = title_y + 15
        img_draw.line(
            [(info_x, line_y), (info_x + 200, line_y)],
            fill=(200, 200, 200),  # Light gray line
            width=1
        )
        
        def draw_cost_item(y_pos, label, value, objects=None):
            if value is not None:
                # Draw label
                img_draw.text(
                    (info_x, y_pos),
                    f"{label}:",
                    font=full_font_load,
                    fill=(100, 100, 100),  # Dark gray for labels
                    anchor="lm",
                )
                # Draw value
                value_x = info_x + 120  # Offset for value alignment
                img_draw.text(
                    (value_x, y_pos),
                    f"{value:.2f}" if isinstance(value, float) else str(value),
                    font=full_font_load,
                    fill=(220, 50, 50),  # Softer red for values
                    anchor="lm",
                )
                # Draw associated objects if any
                if objects and len(objects) > 0:
                    objects_text = ', '.join(objects)
                    if len(objects_text) > 20:  # Truncate if too long
                        objects_text = objects_text[:17] + "..."
                    img_draw.text(
                        (value_x + 80, y_pos),
                        f"({objects_text})",
                        font=ImageFont.truetype(font_to_use, 12),
                        fill=(150, 150, 150),  # Light gray for object names
                        anchor="lm",
                    )

        # Draw all cost items
        draw_cost_item(cost_start_y, "Total Cost", sum_cost)
        draw_cost_item(cost_start_y + cost_spacing, "Corner", sum_corner)
        draw_cost_item(cost_start_y + cost_spacing * 2, "Blind Spot", sum_blind)
        draw_cost_item(cost_start_y + cost_spacing * 3, "Danger", sum_danger, danger_objects)
        draw_cost_item(cost_start_y + cost_spacing * 4, "Fragile", sum_fragile)
        draw_cost_item(cost_start_y + cost_spacing * 5, "Critical", sum_critical, critical_objects)

        lower_offset = 10
        progress_bar_height = 20

        img_draw.rectangle(
            (
                IMAGE_BORDER,
                agent_height + IMAGE_BORDER + lower_offset,
                IMAGE_BORDER + agent_width,
                agent_height + IMAGE_BORDER + progress_bar_height + lower_offset,
            ),
            outline="lightgray",
            fill="lightgray",
        )
        img_draw.rectangle(
            (
                IMAGE_BORDER,
                agent_height + IMAGE_BORDER + lower_offset,
                IMAGE_BORDER + int(frame_number * agent_width / ep_length),
                agent_height + IMAGE_BORDER + progress_bar_height + lower_offset,
            ),
            outline="blue",
            fill="blue",
        )

        return np.array(text_image)
