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
        lib = kwargs["library"]

        for file_path in file_path_lst:
            id = file_path.split('/')[-1].split('.')[-2]

            if lib == "base64":
                with open(file_path, "rb") as image_file:
                    img_dict = {id: base64.b64encode(image_file.read()).decode("utf-8")}
            elif lib == "Pil":
                img_dict = {id: Image.open(file_path)}
            
            yield img_dict
            

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

class PredictionLoader(DataLoaderInterface):
    def __init__(self, *args):
        self.args = args

    def load(self, file_path, **kwargs):
        ex_lst = []
        
        with open(file_path, 'r') as file:
            lines = file.readlines()

            ex_flag = False
            ex_dict = {}
            ex_key = ""
            for l_i, line in tqdm(enumerate(lines), total=len(lines)):
                line = line.strip()

                if "BEGIN:" in line:
                    ex_flag = True
                    ex_id = line.split("[BEGIN: ")[-1]
                    ex_dict["id"] = l_i
                    ex_dict["image_id"] = ex_id.strip(']')
                    continue
                elif "DONE:" in line:
                    ex_id = line.split("[DONE: ")[-1]
                    ex_lst.append(ex_dict)

                    ex_flag = False
                    ex_dict = {}
                    ex_key = ""
                
                if ex_flag:
                    if line == "[Image Topic]":
                        ex_dict["image_topic"] = ""
                        ex_key = "image_topic"
                        continue
                    elif line == "[Animate]":
                        ex_dict["animate"] = []
                        ex_key = "animate"
                        continue
                    elif line == "[Inanimate]":
                        ex_dict["inanimate"] = []
                        ex_key = "inanimate"
                        continue
                    elif line == "[Use or purpose]":
                        ex_dict["use_or_purpose"] = []
                        ex_key = "use_or_purpose"
                        continue
                    elif line == "[Image Description]":
                        ex_dict["image_description"] = ""
                        ex_key = "image_description"
                        continue
                    elif line in ["[Short Answer Question]", "[Short Answer]"]:
                        ex_dict["short_answer"] = {}
                        ex_key = "short_answer"
                        continue
                    elif line in ["[Multiple Choice Question]", "[Multiple Choice]"]:
                        ex_dict["multiple_choice"] = {}
                        ex_key = "multiple_choice"
                        continue
                    elif line in ["[Multiple Select Question]", "[Multiple Select]"]:
                        ex_dict["multiple_select"] = {}
                        ex_key = "multiple_select"
                        continue
                    elif line in ["[True/False Question]", "[True/False"]:
                        ex_dict["true_false"] = {}
                        ex_key = "true_false"
                        continue
                        
                    if line == "":
                        ex_key = ""
                        continue
                    else:
                        if ex_key in ["image_topic", "image_description"]:
                            ex_dict[ex_key] = line
                        elif ex_key == "use_or_purpose":
                            ex_dict[ex_key].append(line)
                        elif ex_key in ["animate", "inanimate"]:
                            if line != "None":
                                ex_dict[ex_key] = line.strip('[]').split(', ')
                        elif ex_key in ["short_answer", "true_false"]:
                            q_split = line.split("(Q) ")
                            a_split = line.split("(A) ")
                            if len(q_split) > 1:
                                ex_dict[ex_key]["question"] = q_split[-1]
                            elif len(a_split) > 1:
                                ex_dict[ex_key]["answer"] = a_split[-1]
                        elif ex_key in ["multiple_choice", "multiple_select"]:
                            q_split = line.split("(Q) ")
                            a_split = line.split("(A) ")
                            if len(q_split) > 1:
                                ex_dict[ex_key]["question"] = q_split[-1]
                            elif len(a_split) > 1:
                                ex_dict[ex_key]["answer"] = a_split[-1].strip(' .')
                            else:
                                if "choice" in ex_dict[ex_key]:
                                    ex_dict[ex_key]["choice"].append(line)
                                else:
                                    ex_dict[ex_key]["choice"] = [line]

        return ex_lst



                        

