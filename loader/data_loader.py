import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import base64


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
            # img = Image.open(file_path)
            id = file_path.split(".")[-2]
                        
            with open(file_path, "rb") as image_file:
                yield {id: base64.b64encode(image_file.read()).decode("utf-8")}

            

class JsonLoader(DataLoaderInterface):
    def __init__(self, *args):
        self.args = args

    def load(self, file_path, **kwargs):
        with open(file_path, 'r') as file:
            try:
                data_lst = json.load(file)
            except:
                lines = file.read()
                data_lst = json.loads(lines)

        return data_lst