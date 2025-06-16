import os
import json
import cv2
import time
import rerun as rr
import rerun.blueprint as rrb
from datetime import datetime
import numpy as np

class RerunEpisodeReader:
    def __init__(self, task_dir = ".", json_file="data.json"):
        self.task_dir = task_dir
        self.json_file = json_file

    def return_episode_data(self, episode_idx):
        # Load episode data on-demand
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_idx:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Episode {episode_idx} data.json not found.")

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            json_file = json.load(jsonf)

        episode_data = []

        # Loop over the data entries and process each one
        for item_data in json_file['data']:
            # Process images and other data
            colors = self._process_images(item_data, 'colors', episode_dir)
            depths = self._process_images(item_data, 'depths', episode_dir)
            audios = self._process_audio(item_data, 'audios', episode_dir)

            # Append the data in the item_data list
            episode_data.append(
                {
                    'idx': item_data.get('idx', 0),
                    'colors': colors,
                    'depths': depths,
                    'states': item_data.get('states', {}),
                    'actions': item_data.get('actions', {}),
                    'tactiles': item_data.get('tactiles', {}),
                    'audios': audios,
                }
            )

        return episode_data

    def _process_images(self, item_data, data_type, dir_path):
        images = {}

        for key, file_name in item_data.get(data_type, {}).items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images[key] = image
        return images

    def _process_audio(self, item_data, data_type, episode_dir):
        audio_data = {}
        dir_path = os.path.join(episode_dir, data_type)

        for key, file_name in item_data.get(data_type, {}).items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    pass  # Handle audio data if needed
        return audio_data

class RerunLogger:
    def __init__(self, prefix = "", IdxRangeBoundary = 30, memory_limit = None):
        self.prefix = prefix
        self.IdxRangeBoundary = IdxRangeBoundary
        rr.init(datetime.now().strftime("Runtime_%Y%m%d_%H%M%S"))
        if memory_limit:
            rr.spawn(memory_limit = memory_limit, hide_welcome_screen = True)
        else:
            rr.spawn(hide_welcome_screen = True)

        # Set up blueprint for live visualization
        if self.IdxRangeBoundary:
            self.setup_blueprint()

    def setup_blueprint(self):
        views = []

        data_plot_paths = [
                           f"{self.prefix}left_arm", 
                           f"{self.prefix}right_arm", 
                           f"{self.prefix}left_hand", 
                           f"{self.prefix}right_hand"
        ]
        for plot_path in data_plot_paths:
            view = rrb.TimeSeriesView(
                origin = plot_path,
                time_ranges=[
                    rrb.VisibleTimeRange(
                        "idx",
                        start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                        end = rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
                plot_legend = rrb.PlotLegend(visible = True),
            )
            views.append(view)

        # Add views for camera images
        image_plot_paths = [
                f"{self.prefix}colors/color_0",
                f"{self.prefix}colors/color_1", 
                f"{self.prefix}colors/color_2",
                f"{self.prefix}colors/color_3"
                ]
        for plot_path in image_plot_paths:
            view = rrb.Spatial2DView(
                origin = plot_path,
                time_ranges=[
                    rrb.VisibleTimeRange(
                        "idx",
                        start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                        end = rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
            )
            views.append(view)

        grid = rrb.Grid(contents = views,
                        grid_columns=4,               # changed from 2 to 4
                        column_shares=[1, 1, 1, 1],   # changed from [1, 1] to [1, 1, 1, 1]
                        row_shares=[1, 1], 
        )
        views.append(rr.blueprint.SelectionPanel(state=rrb.PanelState.Collapsed))
        views.append(rr.blueprint.TimePanel(state=rrb.PanelState.Collapsed))
        rr.send_blueprint(grid)


    def log_item_data(self, item_data: dict):
        try:
            rr.set_time_sequence("idx", item_data.get('idx', 0))

            # Log states
            states = item_data.get('states', {}) or {}
            for part, state_info in states.items():
                if state_info:
                    values = state_info.get('qpos', [])
                    for idx, val in enumerate(values):
                        rr.log(f"{self.prefix}{part}/states/qpos/{idx}", rr.Scalar(val))

            # Log actions
            actions = item_data.get('actions', {}) or {}
            for part, action_info in actions.items():
                if action_info:
                    values = action_info.get('qpos', [])
                    for idx, val in enumerate(values):
                        rr.log(f"{self.prefix}{part}/actions/qpos/{idx}", rr.Scalar(val))

            # Log colors (images)
            colors = item_data.get('colors', {}) or {}
            for color_key, color_val in colors.items():
                # Validate image data before logging
                if color_val is not None and isinstance(color_val, (np.ndarray)) and color_val.ndim in (2, 3):
                    rr.log(f"{self.prefix}colors/{color_key}", rr.Image(color_val))
                elif color_val is not None:
                    if isinstance(color_val, str):
                        # Skip file path strings
                        pass
                    else:
                        print(f"Warning: Invalid image data for {color_key}, shape: {getattr(color_val, 'shape', 'unknown')}")

            # # Log depths (images)
            # depths = item_data.get('depths', {}) or {}
            # for depth_key, depth_val in depths.items():
            #     if depth_val is not None:
            #         # rr.log(f"{self.prefix}depths/{depth_key}", rr.Image(depth_val))
            #         pass # Handle depth if needed

            # # Log tactile if needed
            # tactiles = item_data.get('tactiles', {}) or {}
            # for hand, tactile_vals in tactiles.items():
            #     if tactile_vals is not None:
            #         pass # Handle tactile if needed

            # # Log audios if needed
            # audios = item_data.get('audios', {}) or {}
            # for audio_key, audio_val in audios.items():
            #     if audio_val is not None:
            #         pass  # Handle audios if needed
        except Exception as e:
            print(f"Error in RerunLogger.log_item_data: {e}")

    def log_episode_data(self, episode_data: list):
        for item_data in episode_data:
            self.log_item_data(item_data)
