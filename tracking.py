import numpy as np


class Tracking():

    def __init__(self):
        self.number_of_dections = 0
        self.number_frames_kept = 10
        self.heatmap_list = []
