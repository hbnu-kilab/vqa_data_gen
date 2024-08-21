from tqdm import tqdm
from .data_loader_interface import DataLoaderInterface

class Loader(DataLoaderInterface):
    def __init__(self, 
                 file_system: DataLoaderInterface):
        self.file_system = file_system
    
    def load(self, json_lst, **kwargs):
        return self.file_system.load(json_lst, **kwargs)