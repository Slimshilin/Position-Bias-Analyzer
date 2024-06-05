import os
import warnings
from collections import defaultdict
from typing import List, Dict, Union

from .base import BaseAPI
from subeval.smp import *
import tiktoken
import google.ai.generativelanguage as glm
import google.generativeai as genai

def Gemini_context_window(model):
    length_map = {
        'gemini-1.5-pro': 1048576,
        'gemini-pro': 30720,
        'gemini-pro-vision': 12288,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 30720

class GeminiWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gemini-pro',
                 retry: int = 5,
                 wait: int = 5,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0.9,
                 max_output_tokens: int = 2048,
                 top_p: float = 1.0,
                 top_k: int = 1,
                 **kwargs):

        assert model in [
            'gemini-1.5-pro-latest',
            'gemini-pro',
            'gemini-1.0-pro-latest',
            'gemini-1.0-pro',
            'gemini-1.0-pro-001',
            'gemini-pro-vision',
            'models/embedding-001',
            'models/aqa',
        ]

        self.model = model
        self.cur_idx = 0
        self.fail_cnt = defaultdict(lambda: 0)
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        gemini_keys = []
        if 'KEYS' in os.environ and osp.exists(os.environ['KEYS']):
            keys = load(os.environ['KEYS'])
            gemini_keys = keys.get('gemini-keys', [])

        self.keys = gemini_keys
        self.num_keys = len(self.keys)

        super().__init__(
            wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def generate_inner(self, inputs, **kwargs) -> str:
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'parts': [self.system_prompt]})

        if isinstance(inputs, str):
            messages.append({'role': 'user', 'parts': [inputs]})
        elif isinstance(inputs[0], str):
            messages.extend([{'role': 'user', 'parts': [msg]} for msg in inputs])
        elif isinstance(inputs[0], dict):
            messages.extend(inputs)
        else:
            raise NotImplementedError

        temperature = kwargs.pop('temperature', self.temperature)
        max_output_tokens = kwargs.pop('max_output_tokens', self.max_output_tokens)
        top_p = kwargs.pop('top_p', self.top_p)
        top_k = kwargs.pop('top_k', self.top_k)

        context_window = Gemini_context_window(self.model)
        max_output_tokens = min(max_output_tokens, context_window - self.get_token_len(inputs))
        if 0 < max_output_tokens <= 100:
            warnings.warn('Less than 100 tokens left, may exceed the context window with some additional meta symbols.')
        if max_output_tokens <= 0:
            return 0, self.fail_msg + 'Input string longer than context window.', 'Length Exceeded.'

        for i in range(self.num_keys):
            idx = (self.cur_idx + i) % self.num_keys
            if self.fail_cnt[idx] >= min(self.fail_cnt.values()) + 20:
                continue
            try:
                genai.configure(api_key=self.keys[idx])
                model_obj = genai.GenerativeModel(model_name=self.model)
                response = model_obj.generate_content(
                    glm.Content(
                        parts=[glm.Part(text=m['parts'][0]) for m in messages],
                    ),
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        top_p=top_p,
                        top_k=top_k,
                    ),
                    **kwargs
                )

                result = response.candidates[0].content.parts[0].text.strip()
                self.cur_idx = idx
                return 0, result, 'API Call Succeeded'
            except Exception as e:
                self.fail_cnt[idx] += 1
                if self.verbose:
                    warnings.warn(f'Gemini Key {self.keys[idx]} FAILED !!!')
                    try:
                        warnings.warn(str(e))
                    except:
                        pass

    def get_token_len(self, inputs: Union[str, List[Dict]]) -> int:
        enc = tiktoken.get_encoding("cl100k_base")
        if isinstance(inputs, str):
            return len(enc.encode(inputs))
        else:
            total_tokens = 0
            for msg in inputs:
                total_tokens += len(enc.encode(msg['parts'][0]))
            return total_tokens