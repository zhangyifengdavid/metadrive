from metadrive.obs.state_obs import StateObservation, LidarStateObservation
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.utils import import_pygame
import numpy as np
from collections import deque
import gymnasium as gym


pygame, gfxdraw = import_pygame()
COLOR_WHITE = pygame.Color("white")
DEFAULT_TRAJECTORY_LANE_WIDTH = 3


class BEVLidarObservation(TopDownMultiChannel, LidarStateObservation):
    RESOLUTION = (100, 100)  # pix x pix
    MAP_RESOLUTION = (2000, 2000)  # pix x pix
    CHANNEL_NAMES = ["road_network", "traffic_flow", "navigation", "past_pos"]

    def __init__(self, config):

        vehicle_config = config["vehicle_config"]
        onscreen = config["use_render"]
        clip_rgb = config["norm_pixel"]
        frame_stack = config["frame_stack"]
        post_stack = config["post_stack"]
        frame_skip = config["frame_skip"]
        resolution = (config["resolution_size"], config["resolution_size"])
        max_distance = config["distance"]

        # TopDownObservation class initialization
        super(TopDownMultiChannel, self).__init__(
            vehicle_config, clip_rgb, onscreen=onscreen, resolution=resolution, max_distance=max_distance
        )
        self.num_stacks = 2 + frame_stack
        self.stack_traffic_flow = deque([], maxlen=(frame_stack - 1) * frame_skip + 1)
        self.stack_past_pos = deque(
            [], maxlen=(post_stack - 1) * frame_skip + 1
        )  # In the coordination of target vehicle
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self._should_fill_stack = True
        self.max_distance = max_distance
        self.scaling = self.resolution[0] / max_distance
        assert self.scaling == self.resolution[1] / self.max_distance

        # LidarStateObservation class initialization
        self.state_obs = StateObservation(config)
        super(LidarStateObservation, self).__init__(config)
        self.cloud_points = None
        self.detected_objects = None

    @property
    def observation_space(self):
        """
        env.observation_space["agent_0"] = {'BEV_obs_space': Box(-0.0, 1.0, (224, 224, 5), float32), 'Lidar_obs_space': Box(-0.0, 1.0, (91,), float32)}

        return: a dict of the observation spaces of BEV observation and Lidar observation
        ret_dict["BEV_obs_space"]: gym.spaces.Box(-0.0, 1.0, shape=BEV_shape, dtype=np.float32)
        ret_dict["Lidar_obs_space"]: gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

        """
        ret_dict = {}

        # BEV observation space
        BEV_shape = self.obs_shape + (self.num_stacks,)
        if self.norm_pixel:
            ret_dict["BEV_obs_space"] = gym.spaces.Box(-0.0, 1.0, shape=BEV_shape, dtype=np.float32)
        else:
            ret_dict["BEV_obs_space"] = gym.spaces.Box(0, 255, shape=BEV_shape, dtype=np.uint8)

        # Lidar observation space
        Lidar_shape = list(self.state_obs.observation_space.shape)
        if self.config["vehicle_config"]["lidar"]["num_lasers"] > 0 and self.config["vehicle_config"]["lidar"][
            "distance"] > 0:
            # Number of lidar rays and distance should be positive!
            lidar_dim = self.config["vehicle_config"]["lidar"][
                            "num_lasers"] + self.config["vehicle_config"]["lidar"]["num_others"] * 4
            if self.config["vehicle_config"]["lidar"]["add_others_navi"]:
                lidar_dim += self.config["vehicle_config"]["lidar"]["num_others"] * 4
            Lidar_shape[0] += lidar_dim

        ret_dict["Lidar_obs_space"] = gym.spaces.Box(-0.0, 1.0, shape=tuple(Lidar_shape), dtype=np.float32)
        return ret_dict

    def observe(self, vehicle):
        """
        o["agent_0"]={BEV: np.array with shape (224, 224, 5), Lidar: np.array with shape (91,)}
        """

        o = {}
        # Do not support for multi-agent environment
        # self.render()

        # Get BEV observation
        surface_dict = self.get_observation_window()
        surface_dict["road_network"] = pygame.transform.smoothscale(surface_dict["road_network"], self.resolution)
        img_dict = {k: pygame.surfarray.array3d(surface) for k, surface in surface_dict.items()}

        # Gray scale
        img_dict = {k: self._transform(img) for k, img in img_dict.items()}

        if self._should_fill_stack:
            self.stack_past_pos.clear()
            self.stack_traffic_flow.clear()
            for _ in range(self.stack_traffic_flow.maxlen):
                self.stack_traffic_flow.append(img_dict["traffic_flow"])
            self._should_fill_stack = False
        self.stack_traffic_flow.append(img_dict["traffic_flow"])

        img = [
            img_dict["road_network"] * 2,
            img_dict["past_pos"],
        ]
        indices = self._get_stack_indices(len(self.stack_traffic_flow))

        for i in indices:
            img.append(self.stack_traffic_flow[i])

        # Stack
        img = np.stack(img, axis=2)
        if self.norm_pixel:
            img = np.clip(img, 0, 1.0)
        else:
            img = np.clip(img, 0, 255)

        o["BEV"] = np.transpose(img, (1, 0, 2))

        # Get Lidar observation
        state = self.state_observe(vehicle)
        other_v_info = self.lidar_observe(vehicle)
        self.current_observation = np.concatenate((state, np.asarray(other_v_info)))
        o["Lidar"] = self.current_observation

        return o