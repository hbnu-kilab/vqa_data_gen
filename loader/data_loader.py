import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image


from .data_loader_interface import DataLoaderInterface


class DataLoader(DataLoaderInterface):
    def __init__(self, 
                 file_system: DataLoaderInterface,
                 *args):
        self.file_system = file_system(*args)

    def load(self, file_path, **kwargs):
        return self.file_system.load(file_path, **kwargs)

    def get_listdir(self, root_dir, data_dir):
        data_file_dir = Path(root_dir) / data_dir
        return [str(data_file_dir / data_file) for data_file in os.listdir(data_file_dir)]


class ImageInDirLoader(DataLoaderInterface):
    def __init__(self, *args):
        self.args = args

    def load(self, file_path_lst, **kwargs):
        for file_path in file_path_lst:
            yield Image.open(file_path)
            
