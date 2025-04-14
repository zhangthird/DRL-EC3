import os
import time
import logging


class Setting(object):
    def __init__(self, log):
        self.V = {
            'MAP_X': 16,
            'MAP_Y': 16,
            'MAX_VALUE': 1.,
            'MIN_VALUE': 0.,
            'OBSTACLE': [
                [0, 4, 1, 1],
                [0, 9, 1, 1],
                [0, 10, 2, 1],
                [2, 2, 2, 1],
                [3, 6, 4, 1],
                [4, 4, 1, 4],
                # [4,12, 1, 1],
                [5, 13, 1, 1],
                [6, 12, 2, 1],
                # [10,3, 1, 1],
                [10, 5, 3, 1],
                # [10,9, 1, 1],
                [11, 5, 1, 3],
                [10, 13, 1, 2],
                [11, 13, 2, 1],
                # [11,12,1, 2],
                [12, 0, 1, 2],
                [12, 5, 1, 1],
                [12, 7, 1, 1],
                # [12,13,2, 1],
                [15, 11, 1, 1]
            ],
            'CHANNLE': 3,

            'NUM_UAV': 2,
            'INIT_POSITION': (8, 8),
            'MAX_ENERGY': 500.,
            'NUM_ACTION': 2,
            'RANGE' : 1.1,
            'MAXDISTANCE': 1.,
            'COLLECTION_PROPORTION': 0.2,  # c speed

            'WALL_REWARD': -1.,
            'DATA_REWARD': 1.,
            'WASTE_STEP' : -.5,
            'ALPHA': 1.,
            # 'BETA': 0.01,
            'EPSILON': 1e-4,
            'NORMALIZE': .1,
            'FACTOR': 0.1,
        }
        self.time = str(time.strftime("%Y/%m-%d/%H-%M-%S", time.localtime()))

    def log(self):
        # with open(os.path.join('.', self.time + '.txt'), 'x') as file:
        #     for key, value in self.V.items():
        #         print(key, value, file=file)

        """Logs important settings using the standard logging module."""
        logging.info("--- Environment Settings ---")
        logging.info(f"Setting V: {self.V}")
        # logging.info(f"Setting Dis Penalty Ratio: {self.dis_penalty_ratio}")
        # Add logging for any other important settings initialized in __init__
        # REMOVE: self.LOG.log(self.dis_penalty_ratio)
