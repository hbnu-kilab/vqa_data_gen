import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import openai
from openai import OpenAI

from .promptor_interface import PromptorInterface

class Promptor(PromptorInterface):
    def __init__(self, 
                 file_system: PromptorInterface,
                 *args):
        self.file_system = file_system(*args)
    
    def do_llm(self, instruction):
        return self.file_system.do_llm(instruction)


class ExaonePromptor(PromptorInterface):
    def __init__(self, *args):
        ACCESS_TOKEN = os.environ.get("HFTOKEN")
        login(token=ACCESS_TOKEN)

        self.model = AutoModelForCausalLM.from_pretrained(
            "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")

    def do_llm(self, instruction):
        messages = [
            {"role": "system", 
            "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
            {"role": "user", "content": instruction}
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        output = self.model.generate(
            input_ids.to("cuda"),
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=2048
        )
        
        return self.tokenizer.decode(output[0][len(input_ids[0]):])


class Gemma2Promptor(PromptorInterface):
    def __init__(self, *args):
        # model_id = "rtzr/ko-gemma-2-9b-it"
        model_id = args[0]
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        self.pipeline.model.eval()

    def do_llm(self, instruction):
        messages = [
            {"role": "user", "content": f"{instruction}"}
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        return outputs[0]["generated_text"][len(prompt):]


class ChatGPTPromptor(PromptorInterface):
    def __init__(self, *args):
        ACCESS_TOKEN = os.environ.get("OPENAI_API_KEY")
        
        openai.api_key = ACCESS_TOKEN

        self.client = OpenAI()

        self.model_id = args[0]



    def do_llm(self, instruction):
        messages = [
            {"role": "system", 
            "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction}
            
        ]

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
        )

        return completion.choices[0].message.content
