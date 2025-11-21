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
    # Class variable to track the previous frame's sum_cost
    _previous_sum_cost = None

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
        task_type: str,
        debug: Optional[any],
    ) -> np.array:
        agent_height, agent_width, ch = agent_frame.shape

        font_to_use = "Arial.ttf"  # possibly need a full path here
        full_font_load = ImageFont.truetype(font_to_use, 14)

        IMAGE_BORDER = 25
        TEXT_OFFSET_H = 90  # For action details
        TEXT_OFFSET_V = 30

        # Define two alignment positions - one for labels and one for action details
        action_x = IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H  # For action details
        info_x = IMAGE_BORDER * 2 + agent_width + 20  # For main labels alignment

        task_type_lower = task_type.lower() if task_type else ""
        is_fetch = "fetch" in task_type_lower
        is_pickup = "pickup" in task_type_lower
        right_padding = 380 if is_pickup else (350 if is_fetch else 250)

        image_dims = (
            agent_height + 2 * IMAGE_BORDER + 30,
            agent_width + 2 * IMAGE_BORDER + right_padding,
            ch,
        )
        image = np.full(image_dims, 255, dtype=np.uint8)
        image[
            IMAGE_BORDER : IMAGE_BORDER + agent_height, IMAGE_BORDER : IMAGE_BORDER + agent_width, :
        ] = agent_frame

        # Check if there's any NEW cost in current frame to determine if we need warnings
        sum_cost = debug.get("sum_cost", None)

        # Determine if cost was triggered in the current frame
        # by comparing with previous frame's sum_cost
        has_cost = False
        if sum_cost is not None:
            # If this is the first frame (frame_number == 0), reset previous cost
            if frame_number == 0:
                VideoLogging._previous_sum_cost = 0

            # Check if cost increased from previous frame
            previous_cost = (
                VideoLogging._previous_sum_cost
                if VideoLogging._previous_sum_cost is not None
                else 0
            )
            has_cost = sum_cost > previous_cost

            # Update previous cost for next frame
            VideoLogging._previous_sum_cost = sum_cost

        # Add red border around agent frame if cost is triggered
        if has_cost:
            border_width = 2  # Reduced from 5 to 3 for thinner border
            # Draw red border on the agent frame area
            # Image is in RGB format here, so red is (255, 0, 0)
            cv2.rectangle(
                image,
                (IMAGE_BORDER - border_width, IMAGE_BORDER - border_width),
                (
                    IMAGE_BORDER + agent_width + border_width,
                    IMAGE_BORDER + agent_height + border_width,
                ),
                color=(255, 0, 0),  # Bright red in RGB format (R=255, G=0, B=0)
                thickness=border_width,
            )

            # Add yellow warning triangles on each sensor view (nav and manip)
            # Assuming agent_frame has two sensor views side by side
            sensor_width = agent_width // 2
            warning_size = 30  # Size of the warning triangle

            # Helper function to draw warning triangle
            def draw_warning_triangle(img, x_offset, y_offset, size):
                # Calculate triangle vertices (pointing up)
                x_center = x_offset + size // 2
                y_bottom = y_offset + size
                y_top = y_offset

                # Triangle points
                pt1 = (x_center, y_top)  # Top vertex
                pt2 = (x_offset, y_bottom)  # Bottom left
                pt3 = (x_offset + size, y_bottom)  # Bottom right

                # Note: img is in RGB format (not BGR), so Yellow is (255, 255, 0)
                # Draw filled yellow triangle - RGB format: (R=255, G=255, B=0)
                triangle_points = np.array([pt1, pt2, pt3])

                # Fill with yellow color in RGB
                cv2.fillPoly(img, [triangle_points], color=(255, 255, 0))  # Yellow in RGB

                # Draw black border around triangle
                cv2.polylines(img, [triangle_points], isClosed=True, color=(0, 0, 0), thickness=2)

                # Draw exclamation mark using PIL for better text rendering
                pil_img = Image.fromarray(img)
                draw = ImageDraw.Draw(pil_img)

                # Draw "!" symbol - centered and bold
                font_size = int(size * 0.6)  # Larger font for better visibility
                try:
                    warning_font = ImageFont.truetype(font_to_use, font_size)
                except:
                    warning_font = ImageFont.load_default()

                # Center the exclamation mark properly
                exclamation_x = x_center
                exclamation_y = y_offset + size // 2 + 5  # Center vertically in triangle

                # Draw the exclamation mark multiple times to make it bold/thick
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        draw.text(
                            (exclamation_x + dx, exclamation_y + dy),
                            "!",
                            font=warning_font,
                            fill=(0, 0, 0),  # Black
                            anchor="mm",
                        )

                return np.array(pil_img)

            # Draw warning triangle on navigation view (left sensor, bottom-right corner)
            nav_x = IMAGE_BORDER + sensor_width - warning_size - 10
            nav_y = IMAGE_BORDER + agent_height - warning_size - 10
            image = draw_warning_triangle(image, nav_x, nav_y, warning_size)

            # Draw warning triangle on manipulation view (right sensor, bottom-right corner)
            manip_x = IMAGE_BORDER + agent_width - warning_size - 10
            manip_y = IMAGE_BORDER + agent_height - warning_size - 10
            image = draw_warning_triangle(image, manip_x, manip_y, warning_size)

        text_image = Image.fromarray(image)
        img_draw = ImageDraw.Draw(text_image)
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
            # Define navigation and manipulation actions
            navigation_actions = {
                "move_ahead",
                "move_back",
                "rotate_left",
                "rotate_right",
                "rotate_left_small",
                "rotate_right_small",
                "done",
                "sub_done",
            }
            manipulation_actions = {
                "move_arm_up",
                "move_arm_down",
                "move_arm_in",
                "move_arm_out",
                "move_arm_up_small",
                "move_arm_down_small",
                "move_arm_in_small",
                "move_arm_out_small",
                "wrist_open",
                "wrist_close",
                "pickup",
                "dropoff",
            }

            # Determine task type from task description
            task_type_lower = task_type.lower()
            is_objnav = "objnav" in task_type_lower or "objectnav" in task_type_lower
            is_pickup = "pickup" in task_type_lower
            is_fetch = "fetch" in task_type_lower

            # Separate actions into navigation and manipulation groups
            nav_actions_data = []
            manip_actions_data = []

            if len(action_dist) != len(action_names):
                print(
                    f"Warning: action_dist length ({len(action_dist)}) != action_names length ({len(action_names)})"
                )

            for prob, action in zip(action_dist, action_names):
                try:
                    action_long_name = stretch_long_names[action]
                except KeyError:
                    action_long_name = action

                if action_long_name in navigation_actions:
                    nav_actions_data.append((prob, action, action_long_name))
                elif action_long_name in manipulation_actions:
                    manip_actions_data.append((prob, action, action_long_name))
                else:
                    # Fallback for unknown actions
                    nav_actions_data.append((prob, action, action_long_name))

            if is_fetch:
                action_font_size = 8
                action_spacing = 8
                bar_width = 50
                title_font = ImageFont.truetype(font_to_use, 10)
                action_font = ImageFont.truetype(font_to_use, action_font_size)
            else:
                action_font_size = 9
                action_spacing = 9
                bar_width = 55
                title_font = ImageFont.truetype(font_to_use, 11)
                action_font = ImageFont.truetype(font_to_use, action_font_size)

            # Draw section titles and actions based on task type

            if is_objnav:
                # ObjectNav: only show navigation (8 actions)
                img_draw.text(
                    (action_x, TEXT_OFFSET_V - 15),
                    "Navigation",
                    font=title_font,
                    fill=(50, 50, 150),  # Dark blue
                    anchor="rm",
                )

                # Draw navigation actions
                for i, (prob, action, action_long_name) in enumerate(nav_actions_data):
                    y_pos = TEXT_OFFSET_V + 5 + i * action_spacing
                    img_draw.text(
                        (action_x, y_pos),
                        action_long_name,
                        font=action_font,
                        fill="gray" if action != taken_action else "black",
                        anchor="rm",
                    )
                    img_draw.rectangle(
                        (
                            action_x + 5,
                            y_pos - 1,
                            action_x + 5 + int(bar_width * prob),
                            y_pos + 1,
                        ),
                        outline="blue",
                        fill="blue",
                    )

            elif is_pickup:
                # Pickup: show manipulation in two columns (6+6) to avoid overlapping
                column_spacing = 200
                items_per_column = 6

                pickup_font = ImageFont.truetype(font_to_use, 13)
                pickup_spacing = 15
                pickup_bar_width = 50

                # Draw left column (first 6 actions)
                for i in range(min(items_per_column, len(manip_actions_data))):
                    prob, action, action_long_name = manip_actions_data[i]
                    y_pos = TEXT_OFFSET_V + i * pickup_spacing
                    img_draw.text(
                        (action_x, y_pos),
                        action_long_name,
                        font=pickup_font,
                        fill="gray" if action != taken_action else "black",
                        anchor="rm",
                    )
                    img_draw.rectangle(
                        (
                            action_x + 5,
                            y_pos - 2,
                            action_x + 5 + int(pickup_bar_width * prob),
                            y_pos + 2,
                        ),
                        outline="red",
                        fill="red",
                    )

                # Draw right column (remaining 6 actions)
                manip_x = action_x + column_spacing
                for i in range(items_per_column, len(manip_actions_data)):
                    prob, action, action_long_name = manip_actions_data[i]
                    y_pos = TEXT_OFFSET_V + (i - items_per_column) * pickup_spacing
                    img_draw.text(
                        (manip_x, y_pos),
                        action_long_name,
                        font=pickup_font,
                        fill="gray" if action != taken_action else "black",
                        anchor="rm",
                    )
                    img_draw.rectangle(
                        (
                            manip_x + 5,
                            y_pos - 2,
                            manip_x + 5 + int(pickup_bar_width * prob),
                            y_pos + 2,
                        ),
                        outline="red",
                        fill="red",
                    )

            elif is_fetch:
                # Fetch: show both navigation (left) and manipulation (right) side by side
                column_spacing = 120

                # Navigation title and actions (left column)
                img_draw.text(
                    (action_x, TEXT_OFFSET_V - 15),
                    "Nav",
                    font=title_font,
                    fill=(50, 50, 150),  # Dark blue
                    anchor="rm",
                )

                for i, (prob, action, action_long_name) in enumerate(nav_actions_data):
                    y_pos = TEXT_OFFSET_V + 5 + i * action_spacing
                    short_name = (
                        action_long_name.replace("move_", "")
                        .replace("rotate_", "rot_")
                        .replace("_small", "_s")
                    )
                    img_draw.text(
                        (action_x, y_pos),
                        short_name,
                        font=action_font,
                        fill="gray" if action != taken_action else "black",
                        anchor="rm",
                    )
                    img_draw.rectangle(
                        (
                            action_x + 5,
                            y_pos - 1,
                            action_x + 5 + int(bar_width * prob),
                            y_pos + 1,
                        ),
                        outline="blue",
                        fill="blue",
                    )

                # Manipulation title and actions (right column)
                manip_x = action_x + column_spacing
                img_draw.text(
                    (manip_x, TEXT_OFFSET_V - 15),
                    "Manip",
                    font=title_font,
                    fill=(150, 50, 50),  # Dark red
                    anchor="rm",
                )

                for i, (prob, action, action_long_name) in enumerate(manip_actions_data):
                    y_pos = TEXT_OFFSET_V + 5 + i * action_spacing
                    short_name = action_long_name.replace("move_arm_", "arm_").replace(
                        "_small", "_s"
                    )
                    img_draw.text(
                        (manip_x, y_pos),
                        short_name,
                        font=action_font,
                        fill="gray" if action != taken_action else "black",
                        anchor="rm",
                    )
                    img_draw.rectangle(
                        (
                            manip_x + 5,
                            y_pos - 1,
                            manip_x + 5 + int(bar_width * prob),
                            y_pos + 1,
                        ),
                        outline="red",
                        fill="red",
                    )

            else:
                # Default: show all actions (fallback for unknown task types)
                action_font = ImageFont.truetype(font_to_use, 8)
                action_spacing = 8
                bar_width = 50

                for i, (prob, action) in enumerate(zip(action_dist, action_names)):
                    try:
                        action_long_name = stretch_long_names[action]
                    except KeyError:
                        action_long_name = action

                    y_pos = TEXT_OFFSET_V + 5 + i * action_spacing
                    short_name = action_long_name.replace("move_", "").replace("rotate_", "rot_")
                    img_draw.text(
                        (action_x, y_pos),
                        short_name,
                        font=action_font,
                        fill="gray" if action != taken_action else "black",
                        anchor="rm",
                    )
                    img_draw.rectangle(
                        (
                            action_x + 5,
                            y_pos - 1,
                            action_x + 5 + int(bar_width * prob),
                            y_pos + 1,
                        ),
                        outline="blue",
                        fill="blue",
                    )

        if is_pickup:
            task_info_y = IMAGE_BORDER * 1 + 125
            last_action_y = IMAGE_BORDER * 1 + 145
            error_y = IMAGE_BORDER * 1 + 165
            cost_info_y = IMAGE_BORDER * 1 + 210
        else:
            task_info_y = IMAGE_BORDER * 1 + 110
            last_action_y = IMAGE_BORDER * 1 + 130
            error_y = IMAGE_BORDER * 1 + 150
            cost_info_y = IMAGE_BORDER * 1 + 185

        img_draw.text(
            (info_x, task_info_y),
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
                (info_x, last_action_y),
                "Last Action:",
                font=full_font_load,
                fill=(100, 100, 100),  # Dark gray for consistency
                anchor="lm",
            )
            img_draw.text(
                (info_x + 120, last_action_y),
                "Success" if last_action_success else "Failure",
                font=full_font_load,
                fill=(50, 180, 50) if last_action_success else (220, 50, 50),  # Adjusted colors
                anchor="lm",
            )

        if error_message is not None and error_message != "":
            split_char = "'"
            error_obj = None
            try:
                error_obj = error_message.split(split_char)[1]
            except:
                error_obj = error_message
            img_draw.text(
                (info_x, error_y),
                f"Error: {error_obj}",
                font=full_font_load,
                fill=(220, 50, 50),  # Consistent red color
                anchor="lm",
            )
        # Cost Information Display
        cost_start_y = cost_info_y
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
            [(info_x, line_y), (info_x + 250, line_y)],
            fill=(200, 200, 200),  # Light gray line
            width=1,
        )

        def draw_cost_item(x_pos, y_pos, label, value, label_width=85):
            if value is not None:
                # Draw label
                img_draw.text(
                    (x_pos, y_pos),
                    f"{label}:",
                    font=full_font_load,
                    fill=(100, 100, 100),  # Dark gray for labels
                    anchor="lm",
                )
                # Draw value
                value_x = x_pos + label_width
                img_draw.text(
                    (value_x, y_pos),
                    f"{value:.2f}" if isinstance(value, float) else str(value),
                    font=full_font_load,
                    fill=(220, 50, 50),  # Softer red for values
                    anchor="lm",
                )

        left_col_x = info_x
        right_col_x = info_x + 150

        draw_cost_item(left_col_x, cost_start_y, "Total Cost", sum_cost, label_width=85)
        draw_cost_item(
            left_col_x, cost_start_y + cost_spacing, "Corner", sum_corner, label_width=85
        )
        draw_cost_item(
            left_col_x, cost_start_y + cost_spacing * 2, "Blind Spot", sum_blind, label_width=85
        )

        draw_cost_item(right_col_x, cost_start_y, "Danger", sum_danger, label_width=70)
        draw_cost_item(
            right_col_x, cost_start_y + cost_spacing, "Fragile", sum_fragile, label_width=70
        )
        draw_cost_item(
            right_col_x, cost_start_y + cost_spacing * 2, "Critical", sum_critical, label_width=70
        )

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
