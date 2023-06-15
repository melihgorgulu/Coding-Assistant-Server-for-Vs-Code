from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch


class StarCodeAssistant:
    def __init__(self, _model_name: str = "HuggingFaceH4/starchat-alpha", return_html=False):
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

    def generate_response(self, user_query: str) -> str:
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
                return self.format_response_for_html(self.generate_response(user_query))
            else:
                return self.generate_response(user_query)
        else:
            return "Please give a query to the system."


def test_assistant():
    # model_name = "bigcode/starcoder"
    model_name = "HuggingFaceH4/starchat-alpha"  # Language model that finetuned from StarCoder to act as helpful coding asistant
    query = "write a python function that can calculate cosine similarity between two vector"

    code_assistant = StarCodeAssistant(model_name)
    print(code_assistant)
    print("Used device: ", code_assistant._get_device())
    print("Model response: \n\n")
    print(code_assistant(query))


if __name__ == '__main__':
    test_assistant()


