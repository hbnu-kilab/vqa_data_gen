from tqdm import tqdm
from pathlib import Path
from loader import DataLoader, JsonLoader, ImageInDirLoader, PredictionLoader
from promptor import Promptor, ExaonePromptor, ChatGPTPromptor, LLaVAPromptor
from promptor import mk_inst_for_vqa

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

ROOT_DIR = "/kilab/data/"

model_type = "llava"
do_cda = False
"""
rtzr/ko-gemma-2-9b-it
carrotter/ko-gemma-2b-it-sft
"""
nshot = 0
img_load_lib = "base64"

img_dir = "visual_genome/VG_100K"
img_loader = DataLoader(ImageInDirLoader, "image")
data_dir_list = img_loader.get_listdir(ROOT_DIR, img_dir)
id_img_lst = list(img_loader.load(data_dir_list, library=img_load_lib))  # {"id": img.filename, "image": img}

img_dir = "visual_genome/VG_100K_2"
data_dir_list = img_loader.get_listdir(ROOT_DIR, img_dir)
id_img_lst2 = list(img_loader.load(data_dir_list, library=img_load_lib))  # {"id": img.filename, "image": img}

id_img_lst += id_img_lst2

id_img_dict = {list(el.keys())[0]: el[list(el.keys())[0]] for el in id_img_lst}

pred_path = "./result"
pred_loader = DataLoader(PredictionLoader)
ex_lst = pred_loader.load(pred_path)

if model_type == "exaone":
    model_id = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
    promptor = Promptor(ExaonePromptor, model_id)
elif model_type == "llava":
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    promptor = Promptor(LLaVAPromptor, model_id)
elif model_type in ["gpt-4o-mini", "gpt-4-turbo"]:
    model_id = model_type
    promptor = Promptor(ChatGPTPromptor, model_id)


def baseline(model_type, ex_lst):
    with open(f"./result/pred_vqa_{model_type}", 'w') as pf:
        err_cnt = 0
        for ex in tqdm(ex_lst, total=len(ex_lst)):
            mid = ex["image_id"]
            try:
                img = id_img_dict[mid]
            except:
                err_cnt += 1
                print(f"No ID. {mid}")
                continue
            
            instruction = mk_inst_for_vqa()
            
            output_vqa = promptor.do_llm(instruction, img)
    
            
            print(f"[Output VQA: {mid}]\n{output_vqa}\n")
            print(f"[DONE: {mid}]\n\n")
            pf.write(f"[BEGIN: {mid}]\n{output_vqa}\n[DONE: {mid}]\n\n")

    print(f"ERROR COUNT: {err_cnt}")
            
baseline(model_type, ex_lst)