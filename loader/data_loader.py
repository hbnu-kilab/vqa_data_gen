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
                    if "[Image Topic" in line:
                        ex_dict["image_topic"] = ""
                        ex_key = "image_topic"
                        if line[-1] != ']':
                            line = line.split(']')[-1].strip()
                        else: continue
                    elif "[Animate" in line:
                        ex_dict["animate"] = []
                        ex_key = "animate"
                        if line[-1] != ']':
                            line = line.split(']')[-1].strip()
                        else: continue
                    elif "[Inanimate" in line:
                        ex_dict["inanimate"] = []
                        ex_key = "inanimate"
                        if line[-1] != ']':
                            line = line.split(']')[-1].strip()
                        else: continue
                    elif "[Use or purpose" in line:
                        ex_dict["use_or_purpose"] = []
                        ex_key = "use_or_purpose"
                        if line[-1] != ']':
                            line = line.split(']')[-1].strip()
                        else: continue
                    elif "[Image Description" in line:
                        ex_dict["image_description"] = ""
                        ex_key = "image_description"
                        if line[-1] != ']':
                            line = line.split(']')[-1].strip()
                        else: continue
                    elif "[Short Answer" in line: #[" Question]", "[Short Answer]"]:
                        ex_dict["short_answer"] = {}
                        ex_key = "short_answer"
                        if line[-1] != ']':
                            line = line.split(']')[-1].strip()
                        else: continue
                    elif "[Multiple Choice" in line: # in [ Question]", "[Multiple Choice]"]:
                        ex_dict["multiple_choice"] = {}
                        ex_key = "multiple_choice"
                        if line[-1] != ']':
                            line = line.split(']')[-1].strip()
                        else: continue
                    elif "[Multiple Select" in line: # in [" Question]", "[Multiple Select]"]:
                        ex_dict["multiple_select"] = {}
                        # if len(line.split()) < 4: ex_key = "multiple_select"
                        ex_key = "multiple_select"
                        if line[-1] != ']':
                            line = line.split(']')[-1].strip()
                        else: continue
                    elif "[True/False" in line: #in [ Question]", "[True/False"]:
                        ex_dict["true_false"] = {}
                        ex_key = "true_false"
                        if line[-1] != ']':
                            line = line.split(']')[-1].strip()
                        else: continue
                        
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
                            if "(Q)" in line and "(A)" in line:
                                q_split = line.split("(Q) ")[-1]
                                sub_q_split = q_split.split("(A)")
                                q_split = sub_q_split[0].strip()
                                a_split = sub_q_split[1].strip()
                                ex_dict[ex_key]["question"] = q_split.split(" (True/False)")[0]
                                ex_dict[ex_key]["answer"] = a_split
                            else:
                                q_split = line.split("(Q) ")
                                a_split = line.split("(A) ")

                                if len(q_split) > 1:
                                    ex_dict[ex_key]["question"] = q_split[-1]
                                elif len(a_split) > 1:
                                    ex_dict[ex_key]["answer"] = a_split[-1]
                        elif ex_key in ["multiple_choice", "multiple_select"]:
                            
                            if "(Q)" in line and "(A)" in line:
                                # all elements are listed on one line
                                q_split = line.split("(Q) ")[-1]
                                sub_q_split = q_split.split("(A)")
                                q_split = sub_q_split[0].strip()
                                a_split = sub_q_split[1].strip()
                                q_split, choice = q_split.split('? ')
                                choice = choice.split(') ')
                                ex_dict[ex_key]["question"] = q_split+'?'
                                ex_dict[ex_key]["answer"] = a_split.strip(' .')
                                ex_dict[ex_key]["choice"] = [chr(ord('A')+ch_i) + ') ' + ch[:-2] for ch_i, ch in enumerate(choice[1:-1])] + ["D) "+choice[-1]]
                            else:
                                # should extract element each line
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



                        

