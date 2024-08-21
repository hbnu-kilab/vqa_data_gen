from tqdm import tqdm
from pathlib import Path
from loader import DataLoader, JsonLoader, JsonInDirLoader, SummaryLoader, SummarySBSCLoader, SummarySDSCLoader, SummaryAIHubNewsLoader
from promptor import Promptor, ExaonePromptor, Gemma2Promptor, ChatGPTPromptor
from promptor import mk_inst_for_vqa


from transformers import AutoTokenizer
from eval import eval
from eval.clean_text import clean_data_ko
import evaluate

metric = evaluate.combine(["bleu", "rouge", "meteor"])
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

ROOT_DIR = "/kilab/data/"

model_type = "gpt-4o-mini"
do_cda = False
"""
rtzr/ko-gemma-2-9b-it
carrotter/ko-gemma-2b-it-sft
"""
nshot = 0

data_dir = "etri_images"
data_loader = DataLoader(JsonInDirLoader, "image")
data_dir_list = data_loader.get_listdir(ROOT_DIR, data_dir)
id_img_lst = list(data_loader.load(data_dir_list))  # {"id": img.filename, "image": img}


if model_type == "gemma2":
    model_id = "carrotter/ko-gemma-2b-it-sft"
    # model_id = "rtzr/ko-gemma-2-9b-it"
    promptor = Promptor(Gemma2Promptor, model_id)
elif model_type == "exaone":
    model_id = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
    promptor = Promptor(ExaonePromptor, model_id)
elif model_type in ["gpt-4o-mini", "gpt-4-turbo"]:
    model_id = model_type
    promptor = Promptor(ChatGPTPromptor, model_id)


def baseline(model_type, id_img_lst):
    with open(f"./result/pred_{model_type}", 'w') as pf:
        for id_img in tqdm(id_img_lst, total=len(id_img_lst)):
            id, img = id_img["id"], id_img["image"]
            
            instruction = mk_inst_for_vqa(img)
            
            output_vqa = promptor.do_llm(instruction)
            output_vqa = clean_data_ko(output_vqa)
    
            print(f"[Input Image: {id}] {instruction}")
            print(f"[Output VQA] {output_vqa}\n")
            print("[DONE]")
            pf.write(f"[BEGIN: {id}]\n{output_vqa}\n[DONE: {id}]\n\n")
            
baseline(model_type, id_img_lst)