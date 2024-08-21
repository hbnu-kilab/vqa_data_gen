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
    
    def do_llm(self, instruction, img):
        return self.file_system.do_llm(instruction, img)


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



class ChatGPTPromptor(PromptorInterface):
    def __init__(self, *args):
        ACCESS_TOKEN = os.environ.get("OPENAI_API_KEY")
        
        openai.api_key = ACCESS_TOKEN

        self.client = OpenAI()

        self.model_id = args[0]


    def do_llm(self, instruction, img):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img}"}
                    }  
                ]
            }
        ]

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.0,
        )

        return completion.choices[0].message.content


class LLaVAPromptor(PromptorInterface):
    def __init__(self, *args):
        model_id = args[0]
        self.pipeline = transformers.pipeline("image-to-text", model=model_id)
        self.pipeline.model.eval()

    def do_llm(self, instruction, img):
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image"},
                ],
            },
        ]
        prompt = self.pipeline.tokenizer.apply_chat_template(conversation, add_generation_prompt=True)

        outputs = self.pipeline(img, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

        return outputs[0]["generated_text"][len(prompt):]