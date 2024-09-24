from typing import List
from copy import deepcopy
from tqdm import tqdm
from tqdm.auto import trange
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindnlp.transformers import AutoTokenizer, \
                                AutoModelForCausalLM, \
                                T5ForConditionalGeneration,\
                                BartForConditionalGeneration

class BaseGenerator:
    """`BaseGenerator` is a base object of Generator model."""

    def __init__(self, config):
        self.model_name = config['generator_model']
        self.model_path = config['generator_model_path']

        self.max_input_len = config['generator_max_input_len']
        self.batch_size = config['generator_batch_size']
        self.device = config['device']
        # self.gpu_num =  pynvml.nvmlDeviceGetCount()

        self.generation_params = config['generation_params']

    def generate(self, input_list: list) -> List[str]:
        """Get responses from the generater.

        Args:
            input_list: it contains input texts, each item represents a sample.
        
        Returns:
            list: contains generator's response of each input sample.
        """
        pass

class EncoderDecoderGenerator(BaseGenerator):
    """Class for encoder-decoder model"""

    def __init__(self, config):
        super().__init__(config)
        self.fid = config['use_fid']
        if "t5" in self.model_name:
            if self.fid:
                from flashrag.generator.fid import FiDT5
                self.model = FiDT5.from_pretrained(self.model_path)
            else:
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
        else:
            if self.fid:
                assert False, "FiD only support T5"
            self.model = BartForConditionalGeneration.from_pretrained(self.model_path)
        # self.model.cuda()
        # self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def encode_passages(self, batch_text_passages: List[List[str]]):
        passage_ids, passage_masks = [], []
        for k, text_passages in enumerate(batch_text_passages):
            p = self.tokenizer.batch_encode_plus(
                text_passages,
                max_length=self.max_input_len,
                pad_to_max_length=True,
                return_tensors='ms',
                truncation=True
            )
            passage_ids.append(p['input_ids'][None])
            passage_masks.append(p['attention_mask'][None])

        passage_ids = ops.cat(passage_ids, axis=0)
        passage_masks = ops.cat(passage_masks, axis=0)
        return passage_ids, passage_masks.bool()
    
    def generare(self, input_list: List, batch_size=None, **params):
        if isinstance(input_list, str):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # deal stop params
        stop_sym = None
        if 'stop' in generation_params:
            from flashrag.generator.stop_word_criteria import StopWordCriteria
            stop_sym = generation_params.pop('stop')
            stopping_criteria = [StopWordCriteria(tokenizer=self.tokenizer, prompts=input_list, stop_words=stop_sym)]
            generation_params['stopping_criteria'] = stopping_criteria

        if 'max_tokens' in generation_params:
            if 'max_tokens' in params:
                generation_params['max_new_tokens'] = params.pop('max_tokens')
            else:
                generation_params['max_new_tokens'] = generation_params.pop('max_tokens')

        responses = []
        for idx in trange(0, len(input_list), batch_size, desc='Generation process: '):
            batched_prompts = input_list[idx:idx+batch_size]
            if self.fid:
                # assume each input in input_list is a list, contains K string
                input_ids, attention_mask = self.encode_passages(batched_prompts)
                inputs = {'input_ids': input_ids,
                          'attention_mask': attention_mask}
            else:
                inputs = self.tokenizer(batched_prompts,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True,
                                        max_length=self.max_input_len
                                    )

            # TODO: multi-gpu inference
            outputs = self.model.generate(
                **inputs,
                **generation_params
            )

            outputs = self.tokenizer.batch_decode(outputs,
                                                  skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)

            responses += outputs

        return responses

class HFCausalLMGenerator(BaseGenerator):
    """Class for decoder-only generator, based on hf. """

    def __init__(self, config, model=None):
        super().__init__(config)
        self.config = config
        lora_path = None if 'generator_lora_path' not in config else config['generator_lora_path']
        self.model, self.tokenizer = self._load_model(model=model)
        self.use_lora = False
        if lora_path is not None:
            self.use_lora = True
            self.model.load_adapter(lora_path)

    def _load_model(self, model=None):
        r"""Load model and tokenizer for generator.
        
        """
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                                                        self.model_path,
                                                         ms_dtype="auto",
                                                        #  device_map="auto",
                                                        #  trust_remote_code=True
                                                        )
        # else:
        #     model.cuda()
        # model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if 'qwen' not in self.model_name:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer

    # @torch.inference_mode(mode=True)
    def generate(self, input_list: List[str], batch_size=None, return_scores=False, return_dict=False, **params):
        """Generate batches one by one. The generated content needs to exclude input."""
        if isinstance(input_list, str):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # deal stop params
        stop_sym = None
        if 'stop' in generation_params:
            from flashrag.generator.stop_word_criteria import StopWordCriteria
            stop_sym = generation_params.pop('stop')
            stopping_criteria = [StopWordCriteria(tokenizer=self.tokenizer, prompts=input_list, stop_words=stop_sym)]
            generation_params['stopping_criteria'] = stopping_criteria

        max_tokens = params.pop('max_tokens', None) or params.pop('max_new_tokens', None)
        if max_tokens is not None:
            generation_params['max_new_tokens'] = max_tokens
        else:
            generation_params['max_new_tokens'] = generation_params.get('max_new_tokens', generation_params.pop('max_tokens', None))
        generation_params.pop('max_tokens', None)

        # set eos token for llama
        if 'llama' in self.model_name.lower():
            extra_eos_tokens = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            if 'eos_token_id' in generation_params:
                generation_params['eos_token_id'].extend(extra_eos_tokens)
            else:
                generation_params['eos_token_id'] = extra_eos_tokens

        responses = []
        scores = []
        generated_token_ids = []
        generated_token_logits = []
        self.model.set_grad(requires_grad=False)
        for idx in trange(0, len(input_list), batch_size, desc='Generation process: '):
            # import torch
            # torch.cuda.empty_cache()
            batched_prompts = input_list[idx:idx+batch_size]
            if type(batched_prompts[0]) is list:
                batched_prompts = [prompt[0] for prompt in batched_prompts]
            inputs = self.tokenizer(batched_prompts,
                                    return_tensors="ms",
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_input_len
                                )
            outputs = self.model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                **generation_params
            )


            generated_ids = outputs.sequences
            logits = ops.softmax(ops.stack(outputs.scores, axis=1),axis=-1)
            generated_ids = generated_ids[:, inputs['input_ids'].shape[-1]:]
            gen_score = ops.gather_elements(logits, 2, generated_ids[:, :, None]).squeeze(-1).tolist()
            scores.extend(gen_score)

            # get additinoal info
            if return_dict:
                batch_generated_token_ids = generated_ids
                batch_generated_token_logits = ops.cat([token_scores.unsqueeze(1) for token_scores in outputs.scores], dim=1)
                if batch_generated_token_ids.shape[1] < generation_params['max_new_tokens']:
                    real_batch_size, num_generated_tokens = batch_generated_token_ids.shape 
                    padding_length = generation_params['max_new_tokens'] - num_generated_tokens
                    padding_token_ids = ops.fill(batch_generated_token_ids.dtype,(real_batch_size, padding_length),self.tokenizer.pad_token_id)
                    padding_token_logits = ops.zeros((real_batch_size, padding_length, batch_generated_token_logits.shape[-1]), dtype=batch_generated_token_logits.dtype)
                    batch_generated_token_ids = ops.cat([batch_generated_token_ids, padding_token_ids], dim=1)
                    batch_generated_token_logits = ops.cat([batch_generated_token_logits, padding_token_logits], dim=1)
                generated_token_ids.append(batch_generated_token_ids)
                generated_token_logits.append(batch_generated_token_logits)

            for i, generated_sequence in enumerate(outputs.sequences):
                input_ids = inputs['input_ids'][i]
                text = self.tokenizer.decode(
                            generated_sequence,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                    )
                new_text = text[prompt_length:]

                if stop_sym is not None:
                    strip_stopword = True
                    # Find the first occurrence of any stop word
                    lower_stop_index = len(new_text)  # Default to end of text
                    for sym in stop_sym:
                        stop_index = new_text.find(sym)
                        if stop_index != -1:
                            # Adjust stop index based on whether we're stripping the stop word
                            stop_index += 0 if strip_stopword else len(sym)
                            lower_stop_index = min(stop_index, lower_stop_index)

                    # Cut the text at the first stop word found (if any)
                    new_text = new_text[:lower_stop_index]


                responses.append(new_text.strip())

        if return_dict:
            generated_token_ids = ops.cat(generated_token_ids, axis=0)
            generated_token_logits = ops.cat(generated_token_logits, axis=0)
            return {'generated_token_ids': generated_token_ids, 
                    'generated_token_logits': generated_token_logits,
                    'responses': responses,
                    'scores': scores
                }

        if return_scores:
            return responses, scores
        else:
            return responses

    # @torch.inference_mode(mode=True)
    def cal_gen_probs(self, prev, next):
        input_ids = self.tokenizer.encode(prev, add_special_tokens=False)
        target_ids = self.tokenizer.encode(next, add_special_tokens=False)
        context_ids = input_ids + target_ids
        context_tensor = ms.tensor([context_ids])
        # with torch.no_grad():
        with self.model.set_grad(requires_grad=False):
            outputs = self.model(context_tensor)
            logits = outputs.logits
            logits = logits[0, len(input_ids)-1:len(context_ids)-1, :]
            logits = logits.to(ms.dtype.float32)
            # softmax to normalize
            probs = ops.softmax(logits, axis=-1)
            # obtain probs of target_ids
            target_probs = probs[range(len(target_ids)), target_ids].numpy()

        return logits, target_probs