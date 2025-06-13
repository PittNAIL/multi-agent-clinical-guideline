from typing import Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from litellm import completion
import os

class ModelManager:
    def __init__(self):
        self.models: Dict[str, AutoModelForCausalLM] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        
    def get_model(self, model_name: str, load_in_8bit: bool = True) -> tuple:
        """Get or load a model and its tokenizer"""
        if model_name not in self.models:
            print(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                load_in_8bit=load_in_8bit
            )
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
        
        return self.models[model_name], self.tokenizers[model_name]
    
    async def generate_completion(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        use_litellm: bool = True
    ) -> str:
        """Generate completion using either litellm or direct model call"""
        if use_litellm:
            response = await completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        
        # Direct model call if not using litellm
        model, tokenizer = self.get_model(model_name)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def cleanup(self):
        """Free up GPU memory"""
        for model in self.models.values():
            del model
        self.models.clear()
        torch.cuda.empty_cache() 