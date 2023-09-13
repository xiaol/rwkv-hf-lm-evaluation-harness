import os
from lm_eval.base import BaseLM
from tqdm import tqdm
import time
import json, re, os
import requests
from lm_eval import utils

web = requests.session()
web.headers['Content-Type'] = 'application/x-www-form-urlencoded'

def rwkv_runner_completions(
        client, model, prompt, max_tokens_to_sample, temperature, stop
                            ):
    
    """Query RWKV Runner API for completion.

    Retry with back-off until they respond
    """
    client = 'http://127.0.0.1:8000/v1/completions'

    prompt = f"User: {prompt}\n\nAssistant:"
    # todo text too large.
    params_dict = {
        "prompt": prompt,
        "model": model,
        "stream": False,
        "max_tokens": max_tokens_to_sample,
        "temperature": temperature,
        "top_p": 0.8,
        "presence_penalty": 0.4,
        "frequency_penalty": 0.4,
        "stop":"\n\nUser:",
        "customCuda":True
        }
    
    headers = {'Content-Type': 'application/json'} 
    
    backoff_time = 3
    while True:
        try:
            data = web.post(client, json=params_dict, timeout=600, headers=headers)
            print(data.text)
            result = json.loads(data.text)
            ai_response = result['choices'][0]['text']
            return ai_response

        except Exception as e:
            print('{}'.format(e))
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5



class RWKV_World_LM(BaseLM):
    REQ_CHUNK_SIZE = 20

    def __init__(self, model="RWKV-world-model"):
        """

        :param model: str
        """
        from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
        import rwkv

        super().__init__()

        self.model = model
        self.client = "http://127.0.0.1:8000/v1/completion"

        path = os.path.dirname(rwkv.__file__)
        self.tokenizer = TRIE_TOKENIZER(path+ '\\rwkv_vocab_v20230424.txt')

    @property
    def eot_token_id(self):
        raise NotImplementedError("Not implement yet for RWKV world model.")

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        raise NotImplementedError("No support for logits.")

    def greedy_until(self, requests):
        if not requests:
            return []

        res = []
        for request in tqdm(requests):
            inp = request[0]
            request_args = request[1]
            until = request_args["until"]
            response = rwkv_runner_completions(
                client=self.client,
                model=self.model,
                prompt=inp,
                max_tokens_to_sample=self.max_gen_toks,
                temperature=0.0,
                stop=until,
            )
            res.append(response)
        return res

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
    
