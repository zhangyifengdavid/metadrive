"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
import random

from metadrive import MetaDriveEnv
import argparse

if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        use_render=True,
        manual_control=True,
        traffic_density=0.1,
        environment_num=100,
        random_agent_model=True,
        random_lane_width=True,
        random_lane_num=True,
        map=7,  # seven block
        start_seed=random.randint(0, 1000)
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
    args = parser.parse_args()
    if args.observation == "rgb_camera":
        config.update(dict(offscreen_render=True))
    env = MetaDriveEnv(config)
    env.reset()
    for i in range(1, 1000000000):
        o, r, d, info = env.step([0, 0])
        env.render(text={
            "Auto-Drive (Press T)": env.current_track_vehicle.expert_takeover,
        })
        if d and info["arrive_dest"]:
            env.reset()
    env.close()
