import numpy as np
import redis
from franka_utils.opspace_client import decode_matlab, encode_matlab

from human_shadow.config.redis_keys import *


class Robotiq85Gripper:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        password: str = "",
    ):
        """Initializes the Robotiq85 Gripper Redis client.

        Args:
            host: Redis hostname (the NUC's ip address).
            port: Redis port.
            password: Redis password.
        """
        self._redis = redis.Redis(host=host, port=port, password=password)
        self._redis_pipe = self._redis.pipeline()

    def goto_pose(self, pos: float) -> None:
        """
        Set the opening distance of the Robotiq85 Gripper.
        Args
            pos: int between 0 (fully open) and 1 (fully closed)
        """
        robotiq_pos = (np.array([pos]) * 255).astype(np.uint8)
        self._redis_pipe.set(KEY_ROBOTIQ_CONTROL_COMMAND, encode_matlab(robotiq_pos))
        self._redis_pipe.execute()

    def get_pose(self) -> int:
        """
        Get the opening distance of the Robotiq85 Gripper.
        Returns
            float between 0 (fully open) and 1 (fully closed)
            todo: this is probably wrong - fix
        """
        self._redis_pipe.get(KEY_ROBOTIQ_SENSOR_POS)
        b_pos = self._redis_pipe.execute()
        return decode_matlab(b_pos[0]).item()
