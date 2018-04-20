import os
import numpy as np
import random
import statistics
from preprocess import preprocess
from collaborative_filter import collaborative_filter

class collaborative_filter_bias(collaborative_filter):
    def __init__(self,regCo,K,filepath,density):
        collaborative_filter.__init__(self,regCo,K,filepath,density)