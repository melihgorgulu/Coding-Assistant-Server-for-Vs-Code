
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from transformers import Pipeline, pipeline


# available models: 'starcoder_tiny', 'starchat-alpha', 'bigcode/santacoder', replit/replit-code-v1-3b
class GeneratorBase:
    def generate(self, query: str, parameters: dict) -> str:
        raise NotImplementedError

    def __call__(self, query: str, parameters: dict = None) -> str:
        return self.generate(query, parameters)
    

class StarChat(GeneratorBase):
    def __init__(self, _model_name: str = "HuggingFaceH4/starchat-alpha", return_html=False, device_map: str = None):
        self.model_name = _model_name
        self.return_html = return_html
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(_model_name)
        # to save memory consider using fp16 or bf16 by specifying torch.dtype=torch.float16 for example
        self.model = AutoModelForCausalLM.from_pretrained(_model_name, torch_dtype=torch.float16).to(self.device)
        self.system_prompt = """<|system|>\nBelow is a conversation between a human user and a helpful AI 
        coding assistant. The assistant is happy to help with code questions, and will do its best to understand 
        exactly what is needed. It also tries to avoid giving false or misleading information, 
        and it caveats when it isn’t entirely sure about the right answer.<|end|>\n"""
        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=0,
            eos_token_id=0,
            min_new_tokens=32,
            max_new_tokens=512,
            early_stopping=True
        )

    def __str__(self):
        return f"Code assistant with model {self.model_name}"

    def generate_prompt(self, user_query: str) -> str:
        user_prompt = f"<|user|>\n{user_query}<|end|>\n"

        assistant_prompt = "<|assistant|>"

        full_prompt = self.system_prompt + user_prompt + assistant_prompt
        return full_prompt

    def generate(self, user_query: str, paremeters) -> str:
        prompt = self.generate_prompt(user_query)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs, generation_config=self.generation_config)
        model_output = self.tokenizer.decode(outputs[0])
        model_output = model_output[len(prompt):]
        if "<|end|>" in model_output:
            cut = model_output.find("<|end|>")
            model_output = model_output[:cut]
        return model_output

    def _get_device(self):
        return self.device

    def format_response_for_html(self, response: str):
        response_html = response.replace("\n", "<br>")
        return response_html

    def __call__(self, user_query: str):
        if user_query:
            if self.return_html:
                return self.format_response_for_html(self.generate(user_query))
            else:
                return self.generate(user_query)
        else:
            return "Please give a query to the system."



class StarCoder(GeneratorBase):
    def __init__(self, _model_name: str = "bigcode/starcoder", device_map: str = None):
        self.model_name: str = _model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipe: Pipeline = pipeline(
            "text-generation", model=_model_name, torch_dtype=torch.bfloat16, device=self.device, 
            device_map=device_map)
        
        self.generation_config = GenerationConfig.from_pretrained(_model_name)
        self.generation_config.pad_token_id = self.pipe.tokenizer.eos_token_id

    def generate(self, query: str, parameters: dict) -> str:
        config: GenerationConfig = GenerationConfig.from_dict({
            **self.generation_config.to_dict(),
            **parameters
        })
        json_response: dict = self.pipe(query, generation_config=config)[0]
        generated_text: str = json_response['generated_text']
        return generated_text
    
    def set_config(self, **kwargs):
        generation_config_dict = self.generation_config.to_dict() 
        for key, value in kwargs.items():
            generation_config_dict[f"{key}"] = value
        self.generation_config = GenerationConfig.from_dict(generation_config_dict)
        
class SantaCoder(GeneratorBase):
    def __init__(self, _model_name: str = "bigcode/santacoder", device_map: str = None):
        self.model_name: str = _model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(_model_name, trust_remote_code=True)
        self.model.to(device=self.device)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(_model_name, trust_remote_code=True)
        self.generation_config: GenerationConfig = GenerationConfig.from_model_config(self.model.config)
        self.generation_config.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, query: str, parameters: dict) -> str:
        input_ids: torch.Tensor = self.tokenizer.encode(query, return_tensors='pt').to(self.device)
        config: GenerationConfig = GenerationConfig.from_dict({
            **self.generation_config.to_dict(),
            **parameters
        })
        output_ids: torch.Tensor = self.model.generate(input_ids, generation_config=config)
        output_text: str = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text
