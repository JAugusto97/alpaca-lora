import torch
import sys
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from typing import List

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

class AlpacaLora:
    def __init__(
        self,
        base_model="decapoda-research/llama-7b-hf",
        lora_weights="tloen/alpaca-lora-7b",
        load_8bit=True,
    ):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        try:
            if torch.backends.mps.is_available():
                self.device = "mps"
        except:  # noqa: E722
            pass

        self.load_model(base_model, lora_weights, load_8bit)

    def evaluate(
        self,
        instruction,
        input,
        temperature,
        top_p,
        top_k,
        num_beams,
        max_new_tokens,
        stream_output=False,
        **kwargs,
    ):
        prompt = self.prompter.generate_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    self.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = self.tokenizer.decode(output)

                    if output[-1] in [self.tokenizer.eos_token_id]:
                        break

                    yield self.prompter.get_response(decoded_output)
            return  # early return for stream_output

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
            s = generation_output.sequences[0]
            output = self.tokenizer.decode(s)
            yield self.prompter.get_response(output)

    def load_model(
        self,
        base_model="decapoda-research/llama-7b-hf",
        lora_weights="tloen/alpaca-lora-7b",
        load_8bit=True,
    ):
        prompt_template: str = ""
        self.prompter = Prompter(prompt_template)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        self.model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_weights,
            torch_dtype=torch.float16,
        )

        # unwind broken decapoda-research config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        if not load_8bit:
            self.model.half()  # seems to fix bugs for some users.

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

    def prompt(
        self,
        prompt: str,
        temperature=0.1,
        top_k=50,
        top_p=0.75,
        num_beams=4,
        max_new_tokens=512,
    ):
        answer = ""
        for item in self.evaluate(
            prompt,
            input=None,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        ):
            answer += item

        return answer
