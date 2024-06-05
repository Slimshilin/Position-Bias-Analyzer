import os
import warnings
from collections import defaultdict
from typing import List, Dict, Union

from .base import BaseAPI
from subeval.smp import *
import tiktoken
import anthropic

def Claude_context_window(model):
    length_map = {
        'claude-3-opus-20240229': 200000,
        'claude-3-sonnet-20240229': 200000,
        'claude-3-haiku-20240307': 200000,
        'claude-2.1': 200000,
        'claude-2.0': 100000,
        'claude-instant-1.2': 100000,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 100000

class ClaudeWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'claude-3-opus-20240229',
                 retry: int = 5,
                 wait: int = 5,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 1,
                 max_tokens: int = 1024,
                 **kwargs):

        assert model in [
            'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307',
            'claude-2.1', 'claude-2.0', 'claude-instant-1.2'
        ]

        self.model = model
        self.cur_idx = 0
        self.fail_cnt = defaultdict(lambda: 0)
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature

        claude_keys = []
        if 'KEYS' in os.environ and osp.exists(os.environ['KEYS']):
            keys = load(os.environ['KEYS'])
            claude_keys = keys.get('claude-keys', [])

        self.keys = claude_keys
        self.num_keys = len(self.keys)

        super().__init__(
            wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def generate_inner(self, inputs, **kwargs) -> str:
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        if isinstance(inputs, str):
            messages.append({"role": "user", "content": inputs})
        elif isinstance(inputs[0], str):
            for msg in inputs:
                messages.append({"role": "user", "content": msg})
        elif isinstance(inputs[0], dict):
            messages.extend(inputs)
        else:
            raise NotImplementedError

        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        context_window = Claude_context_window(self.model)
        max_tokens = min(max_tokens, context_window - self.get_token_len(inputs))
        if 0 < max_tokens <= 100:
            warnings.warn('Less than 100 tokens left, may exceed the context window with some additional meta symbols.')
        if max_tokens <= 0:
            return 0, self.fail_msg + 'Input string longer than context window.', 'Length Exceeded.'

        for i in range(self.num_keys):
            idx = (self.cur_idx + i) % self.num_keys
            if self.fail_cnt[idx] >= min(self.fail_cnt.values()) + 20:
                continue
            try:
                c = anthropic.Anthropic(api_key=self.keys[idx])
                response = c.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                    **kwargs
                )

                result = response.content[-1].text.strip()
                self.cur_idx = idx
                return 0, result, 'API Call Succeeded'
            except Exception as e:
                self.fail_cnt[idx] += 1
                if self.verbose:
                    warnings.warn(f'Claude Key {self.keys[idx]} FAILED !!!')
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
                total_tokens += len(enc.encode(msg['content']))
            return total_tokens