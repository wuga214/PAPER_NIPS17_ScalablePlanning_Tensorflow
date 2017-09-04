import os
import pandas as pd

#Given local path, find full path
def PathFinder(path):
    # python 2
    script_dir = os.path.dirname('__file__')
    full_path = os.path.join(script_dir, path)
    # python 3
    # full_path=os.path.abspath(path)
    return full_path


#Read Data for Deep Learning
def ReadData(path):
    full_path=PathFinder(path)
    return pd.read_csv(full_path, sep=',', header=0)