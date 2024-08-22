from tqdm import tqdm
from pathlib import Path
from loader import DataLoader, JsonLoader, ImageInDirLoader, PredictionLoader
from promptor import Promptor, ExaonePromptor, ChatGPTPromptor, LLaVAPromptor
from promptor import mk_vqa_for_multiple_choice

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

ROOT_DIR = "/kilab/data/"

model_type = "gpt-4o-mini"
do_cda = False
"""
gpt-4o-mini
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

pred_path = "./result/pred_gpt-4o-mini.0822"
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
        choice_cnt, exact_cnt = 0, 0
        for ex in tqdm(ex_lst, total=len(ex_lst)):
            mid = ex["image_id"]
            try:
                img = id_img_dict[mid]
            except:
                err_cnt += 1
                print(f"No ID. {mid}")
                continue
            
            mc_question = ex["multiple_choice"]["question"]
            mc_choice = ex["multiple_choice"]["choice"]
            mc_answer = ex["multiple_choice"]["answer"]
            instruction = mk_vqa_for_multiple_choice(mc_question, mc_choice)
            
            output_vqa = promptor.do_llm(instruction, img)

            if '(A)' in output_vqa:
                pred_ans = output_vqa.split('(A)')[-1].strip(' \n(.')
            else:
                pred_ans = output_vqa.split('[Multiple Choice]')[-1].strip(' \n(.')
            
            if pred_ans[:2] == mc_answer[:2]:
                choice_cnt += 1
            if pred_ans == mc_answer:
                exact_cnt += 1
            
            print(f"[EX-BEGIN: {mid}]\nQUESTION: {mc_question}\nANSWER: {mc_answer}")
            print(f"PRED_ANSWER: {pred_ans}\nEXACT_ANSWER: {exact_cnt}\tCHOICE_ANSWER: {choice_cnt}\n[DONE: {mid}]\n\n")
            pf.write(f"[EX-BEGIN: {mid}]\nQUESTION: {mc_question}\nANSWER: {mc_answer}")
            pf.write(f"PRED_ANSWER: {pred_ans}\nEXACT_ANSWER: {exact_cnt}\tCHOICE_ANSWER: {choice_cnt}\n[DONE: {mid}]\n\n")
            pf.write(f"<LOG>{output_vqa}</LOG>")

    print(f"SCORE: {choice_cnt/len(ex_lst)}")
    print(f"EXACT SCORE: {exact_cnt/len(ex_lst)}")
    print(f"ERROR COUNT: {err_cnt}")
            

def post_proc(output):
    for line in output:
        line = line.strip()


baseline(model_type, ex_lst)