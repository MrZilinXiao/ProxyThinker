# python-implementation of Base + \alpha (RL - Base)
from typing import Optional, Dict, Any, List, Sequence
import torch
from transformers import (
    Qwen2VLForConditionalGeneration, 
    GenerationMixin
)
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessorList, 
    NoBadWordsLogitsProcessor, 
    SuppressTokensAtBeginLogitsProcessor
)
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from qwen_vl_utils import process_vision_info


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)
    

# base model image loader -- for Instruct model use qwen_vl_utils
def load_images(messsages):
    images = []
    for message in messsages:
        for item in message:
            if item['type'] == 'image':
                if type(item['image']) == str:
                    image_path = item['image']
                    image = Image.open(image_path)
                    images.append(image)
                else:
                    images.append(item['image'])
    return images


def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class ProxyThinkerWrapper(GenerationMixin):
    # Tests pending with Qwen2-VL -- once passed it should be compatible with all Qwen-VL
    # highly optimized compared to official contrastive decoding and proxytuning
    def __init__(
        self, 
        base_model: Qwen2VLForConditionalGeneration,
        preprocessor,  # tokenizer is in preprocessor.tokenizer
        positive_model: Optional[Qwen2VLForConditionalGeneration] = None,
        negative_model: Optional[Qwen2VLForConditionalGeneration] = None,
        do_torch_compile: Optional[bool] = False,
        alpha: Optional[float] = 1.0, 
        input_device_dict: Optional[Dict[str, torch.device]] = None,
        logits_device: Optional[torch.device] = None,
    ):
        self.base_model = base_model
        self.positive_model = positive_model
        self.negative_model = negative_model
        self.preprocessor = preprocessor
        self.alpha = alpha
        
        self.base_model.eval()
        if self.positive_model is not None:
            self.positive_model.eval()
        
        if self.negative_model is not None:
            self.negative_model.eval()
        
        if do_torch_compile:
            print("Using torch.compile() for models... Expect some abnormal behavior")
            self.base_model = torch.compile(self.base_model)
            if self.positive_model is not None:
                self.positive_model = torch.compile(self.positive_model)
            if self.negative_model is not None:
                self.negative_model = torch.compile(self.negative_model)
        
        self.tokenizer = preprocessor.tokenizer
        
        self.input_device_dict = input_device_dict
        if self.input_device_dict is None:
            self.input_device_dict = {
                'base': self.base_model.model.embed_tokens.weight.device,
                'positive': self.positive_model.model.embed_tokens.weight.device if self.positive_model is not None else None,
                'negative': self.negative_model.model.embed_tokens.weight.device if self.negative_model is not None else None
            }
        
        self.logits_device = logits_device
        if self.logits_device is None:
            # logits_device defaults to where lm_head sits
            self.logits_device = self.base_model.lm_head.weight.device
    
    def update_analysis_data(self, analysis_data, next_tokens, next_token_logits_dict):
        analysis_data['tokens'].append([self.tokenizer.decode(t) for t in next_tokens])
        analysis_data['token_ids'].append(next_tokens)

        # logits from each model for the next token
        for model in next_token_logits_dict.keys():
            analysis_data[f'logits_{model}'].append(next_token_logits_dict[model].unsqueeze(dim=1))

        return analysis_data
            
    def forward(self, base_inputs, positive_inputs, negative_inputs, return_dict=False):
        base_outputs = self.base_model(**base_inputs, return_dict=return_dict)
        positive_outputs, negative_outputs = None, None
        if self.positive_model is not None:
            positive_outputs = self.positive_model(**positive_inputs, return_dict=return_dict)
        
        if self.negative_model is not None:
            negative_outputs = self.negative_model(**negative_inputs, return_dict=return_dict)
        
        return base_outputs, positive_outputs, negative_outputs
    
    def generate(
        self, 
        inputs,  # input_ids, attention_mask, pixel_values, image_grid_thw
        positive_inputs=None, 
        negative_inputs=None, 
        max_new_tokens: Optional[int] = 1024,
        do_sample: bool = False,
        top_p: float = 1.0,
        top_k: int = 0,
        temperature: float = 0.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        return_logits_for_analysis: bool = False,
        **kwargs
    ):
        base_kwargs = kwargs.copy()
        positive_kwargs = kwargs.copy()
        negative_kwargs = kwargs.copy()
        
        inputs = inputs.to(self.input_device_dict['base'])
        if positive_inputs is None and self.positive_model is not None:
            positive_inputs = deepcopy(inputs).to(self.input_device_dict['positive'])
        
        if negative_inputs is None and self.negative_model is not None:
            negative_inputs = deepcopy(inputs).to(self.input_device_dict['negative'])
            
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(inputs.input_ids.shape[0], dtype=torch.long, device=inputs.input_ids.device)
        # TODO: how about multiple stop ids?
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(inputs.input_ids.device)
        
        if return_logits_for_analysis:
            analysis_data = defaultdict(list)
            
        # added: before generation, init_cache_position to cumulated token [0, 1, 2, ...]
        base_kwargs = self._get_initial_cache_position(inputs.input_ids, base_kwargs)
        if positive_inputs is not None:
            positive_kwargs = self._get_initial_cache_position(positive_inputs.input_ids, positive_kwargs)
        if negative_inputs is not None:
            negative_kwargs = self._get_initial_cache_position(negative_inputs.input_ids, negative_kwargs)
            
        # move attention mask from inputs to kwargs, so that it will be updated via `_update_model_kwargs_for_generation`
        if "attention_mask" in inputs:
            base_kwargs["attention_mask"] = inputs.pop("attention_mask")
        
        if positive_inputs is not None and "attention_mask" in positive_inputs:
            positive_kwargs["attention_mask"] = positive_inputs.pop("attention_mask")
            
        if positive_inputs is not None and "attention_mask" in negative_inputs:
            negative_kwargs["attention_mask"] = negative_inputs.pop("attention_mask")
            
        for step in range(max_new_tokens):
            # use_cache effective here, also convert attention_mask to 4d
            # print(f"base_kwargs: {base_kwargs.keys()}")
            # print(f"inputs: {inputs.keys()}")  # why inputs gets cache_position somewhere?
            # let's get a joint set and assert if they are the same
            
            # def filter_inputs(inputs, kwargs):
            #     common_keys = set(kwargs.keys()).intersection(set(inputs.keys()))
            #     for k in common_keys:
            #         # unfortunately they are not the same
            #         # if kwargs[k] != inputs[k]:
            #         #     print(f"{step}: base_kwargs and inputs should have the same {k}, but got {base_kwargs[k]} and {inputs[k]}")
            #         #     pdb.set_trace()
            #         # remove common keys from inputs
            #         inputs.pop(k)

            # filter_inputs(inputs, base_kwargs)
            base_model_inputs = self.base_model.prepare_inputs_for_generation(**inputs, **base_kwargs)
            positive_model_inputs, negative_model_inputs = None, None
            
            if self.positive_model is not None:
                # filter_inputs(positive_inputs, positive_kwargs)
                positive_model_inputs = self.positive_model.prepare_inputs_for_generation(**positive_inputs, **positive_kwargs)
            
            if self.negative_model is not None:
                # filter_inputs(negative_inputs, negative_kwargs)
                negative_model_inputs = self.negative_model.prepare_inputs_for_generation(**negative_inputs, **negative_kwargs)
                
            # this prepare_inputs_for_generation() put inputs to dict data type
            
            # forward pass three models
            base_outputs, positive_outputs, negative_outputs = self.forward(
                base_model_inputs, positive_model_inputs, negative_model_inputs, return_dict=True
            )
            
            # get last token logits
            base_next_token_logits = base_outputs.logits[:, -1, :]
            
            if positive_outputs is not None:
                positive_next_token_logits = positive_outputs.logits[:, -1, :]
            
            if negative_outputs is not None:
                negative_next_token_logits = negative_outputs.logits[:, -1, :]
                
            do_trucate_small = False
            larger_vocab_size = base_next_token_logits.shape[-1]
            # at least positive and negative logits should be of the same shape
            if positive_outputs is not None and negative_outputs is not None:   # go to proxytuning branch
                assert positive_next_token_logits.shape == negative_next_token_logits.shape, \
                    f"positive and negative logits should have the same shape, but got {positive_next_token_logits.shape} and {negative_next_token_logits.shape}"
            
                # trucate logits to have the same shape
                larger_vocab_size = max(
                    base_next_token_logits.shape[-1], positive_next_token_logits.shape[-1]
                )

                if base_next_token_logits.shape[-1] != larger_vocab_size:
                    # base is smaller -- truncate positive and negative
                    do_trucate_small = True
                    positive_next_token_logits = positive_next_token_logits[:, :base_next_token_logits.shape[-1]]
                    negative_next_token_logits = negative_next_token_logits[:, :base_next_token_logits.shape[-1]]
                else:  # base is equal or larger -- truncate base
                    base_next_token_logits = base_next_token_logits[:, :positive_next_token_logits.shape[-1]]
                    
                assert base_next_token_logits.shape == positive_next_token_logits.shape
                positive_next_token_logits = positive_next_token_logits[:, :base_next_token_logits.shape[-1]]
                
                # before logits processing, make sure all logits are on the same device
                base_next_token_logits = base_next_token_logits.to(self.logits_device)
                positive_next_token_logits = positive_next_token_logits.to(self.logits_device)
                negative_next_token_logits = negative_next_token_logits.to(self.logits_device)
                
                next_token_logits = (
                    base_next_token_logits + 
                    self.alpha * (positive_next_token_logits - negative_next_token_logits)
                )
                
            else:  # normal decoding branch
                next_token_logits = base_next_token_logits
                
            next_token_logits = next_token_logits.to(inputs['input_ids'].device)
            
            if logits_processor is not None:
                next_token_logits = logits_processor(inputs['input_ids'], next_token_logits)
                
            if temperature != 0.0:
                next_token_logits = next_token_logits / temperature
                
            if top_p < 1.0 or top_k > 0:
                next_token_logits = top_k_top_p_filtering(
                    next_token_logits, top_k=top_k, top_p=top_p
                )
                
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                if not do_trucate_small:
                    if positive_outputs is not None and negative_outputs is not None:
                        probs = torch.cat([
                            probs, 
                            torch.zeros(
                                (probs.shape[0], larger_vocab_size - probs.shape[1]), 
                                device=probs.device
                            )
                        ], dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
            # sampling happens on gpu:0
            # next_tokens = next_tokens.to(inputs['input_ids'].device)
            next_tokens = (
                next_tokens * unfinished_sequences + 
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )
            
            if return_logits_for_analysis:
                next_token_logits_dict = {
                    'proxy': next_token_logits,
                    'base': base_next_token_logits,
                    'positive': positive_next_token_logits,
                    'negative': negative_next_token_logits
                }
                analysis_data = self.update_analysis_data(
                    analysis_data, next_tokens, next_token_logits_dict
                )

            inputs['input_ids'] = torch.cat([inputs['input_ids'], 
                                            next_tokens.unsqueeze(1).to(self.input_device_dict['base'])], 
                                            dim=-1)
            
            if positive_inputs is not None:
                positive_inputs['input_ids'] = torch.cat([positive_inputs['input_ids'], 
                                                          next_tokens.unsqueeze(1).to(self.input_device_dict['positive'])], dim=-1)
            
            if negative_inputs is not None:
                negative_inputs['input_ids'] = torch.cat([negative_inputs['input_ids'], 
                                                          next_tokens.unsqueeze(1).to(self.input_device_dict['negative'])], dim=-1)
            
            # update kwargs -- copy past_key_values and extend attention mask
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            # this: 
            # 1. removes long cache_position and put into one value
            # 2. copy past_key_values to kwargs
            # 3. extend attention mask if provided
            if positive_outputs is not None:
                positive_kwargs = self._update_model_kwargs_for_generation(positive_outputs, positive_kwargs)
            
            if negative_outputs is not None:
                negative_kwargs = self._update_model_kwargs_for_generation(negative_outputs, negative_kwargs)
            
            # breakpoint()
            
            if stopping_criteria is not None and torch.all(
                stopping_criteria(inputs['input_ids'], None
            )):
                # stop when all sequences finished with stop ids
                break
            
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            
            if unfinished_sequences.max() == 0:
                # all sequences finished
                break
            
        if return_logits_for_analysis:
            for k in analysis_data.keys():
                if k.startswith('logits'):
                    analysis_data[k] = torch.cat(analysis_data[k], dim=1)
            return inputs['input_ids'], analysis_data
        
        # breakpoint()  # check why it's not finishing
        return inputs['input_ids']
            
    # deprecated, let's use GenerationMixin._update_model_kwargs_for_generation
    # def _update_model_kwargs_for_generation(
    #     self,
    #     outputs: ModelOutput,
    #     kwargs: Dict[str, Any],
    # ) -> Dict[str, Any]:
    #     # update past_key_values
    #     kwargs["past_key_values"] = outputs.past_key_values

    #     # update attention mask
    #     if "attention_mask" in kwargs:
    #         attention_mask = kwargs["attention_mask"]
    #         kwargs["attention_mask"] = torch.cat(
    #             [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
    #         )

    #     return kwargs
    
@torch.inference_mode()
def generate_completions(
    # args, 
    model_wrapper, 
    processor,  # could we share the processor? I guess so... 
    # both tokenizer and image preprocessor are in processor
    messages,  # List of Dict -- for base model it was single message w/ image
    positive_messages: Optional[List[Dict]] = None,
    negative_messages: Optional[List[Dict]] = None,
    batch_size=8, 
    stop_id_seqs: Optional[List[List[int]]] = None,  # try not to apply this...
    banned_id_seqs: Optional[List[List[int]]] = None,  # never sample these ids during generation
    banned_begin_ids: Optional[List[int]] = None,  # never sample these ids at the beginning -- usually suppress EOS
    disable_tqdm: Optional[bool] = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0, 
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    use_cache: bool = True,
    return_logits_for_analysis: bool = False, 
    is_instruct: bool = False,
    **generation_kwargs, # other config e.g. max_new_tokens, do_sample, etc.
):
    # hack qwen2.5vl
    processor.tokenizer.padding_side = "left"
    
    generations = []
    
    if not disable_tqdm:
        progress = tqdm(total=len(messages), desc="Generating")
    
    if positive_messages is not None:
        assert len(messages) == len(positive_messages),\
            f"messages and positive_messages should have the same length, but with {len(messages)} and {len(positive_messages)}"
    
    if negative_messages is not None: 
        assert len(messages) == len(negative_messages),\
            f"messages and negative_messages should have the same length"
        
    num_return_sequences = generation_kwargs.get('num_return_sequences', 1)
    assert num_return_sequences == 1, "num_return_sequences > 1 is not supported yet"
    
    stopping_criteria = None
    if stop_id_seqs is not None:
        stopping_criteria = StoppingCriteriaList([
            KeyWordsCriteria(stop_id_sequences=stop_id_seqs)
        ])
    
    for i in range(0, len(messages), batch_size):
        batch_messages = messages[i:i + batch_size]
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        if is_instruct:
            images, _ = process_vision_info(batch_messages)
        else:
            images = load_images(batch_messages)
        
        positive_inputs = None
        negative_inputs = None
        
        if positive_messages is not None:
            positive_text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in positive_messages[i:i + batch_size]]
            positive_inputs = processor(
                text=positive_text,
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(model_wrapper.input_device_dict['positive'])
            
        if negative_messages is not None:
            negative_text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in negative_messages[i:i + batch_size]]
            negative_inputs = processor(
                text=negative_text,
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(model_wrapper.input_device_dict['negative'])
        
        inputs = processor(
            text=text,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(model_wrapper.input_device_dict['base'])
        
        # create logit processor per batch as it is varied per batch
        # only applies to input.input_ids 
        logit_processors = []
        if banned_id_seqs is not None:
            logit_processors.append(
                NoBadWordsLogitsProcessor(
                    banned_id_seqs, eos_token_id=processor.tokenizer.eos_token_id
                )
            )
        if banned_begin_ids is not None:
            logit_processors.append(
                SuppressTokensAtBeginLogitsProcessor(
                    banned_begin_ids, begin_index=inputs.input_ids.shape[1], 
                    device=inputs.input_ids.device
                )
            )
        logits_processor = None
        if logit_processors:
            logits_processor = LogitsProcessorList(logit_processors)
            
        # before putting into model, record the input_ids length
        input_ids_len = inputs.input_ids.shape[1]
        
        batch_output_ids = model_wrapper.generate(
            inputs,
            positive_inputs, 
            negative_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            return_logits_for_analysis=return_logits_for_analysis,
            # --- below passed to kwargs ---
            use_cache=use_cache,
            **generation_kwargs
        )
        if return_logits_for_analysis:
            batch_output_ids, analysis_data = batch_output_ids
        
        # remove after stop_id_seqs -- but I think it's not needed
        if stop_id_seqs is not None:
            for output_idx in range(batch_output_ids.shape[0]):
                # for token_idx in range(inputs.input_ids.shape[1], batch_output_ids.shape[1]):
                for token_idx in range(input_ids_len, batch_output_ids.shape[1]):
                    if any(batch_output_ids[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_seqs):
                        batch_output_ids[output_idx, token_idx:] = processor.tokenizer.pad_token_id
                        break
                    
        # batch_output_ids_trimmed = [
        #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, batch_output_ids)
        # ]
        batch_output_ids_trimmed = batch_output_ids[:, input_ids_len:]
        
        batch_output_text = processor.batch_decode(
            batch_output_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        generations += batch_output_text
        
        if not disable_tqdm:
            progress.update(len(batch_messages) // num_return_sequences)
            
        # print at some interval to see the progress
        if (i // batch_size) % 1 == 0:
            # breakpoint()
            print(f"Batch {i // batch_size}: {batch_output_text[0]}")
            # inputs_text = processor.batch_decode(
            #     inputs.input_ids[0, :input_ids_len], skip_special_tokens=True,
            #     clean_up_tokenization_spaces=False
            # )
            # print(f"Batch {i // batch_size} inputs: {inputs_text}")
            
    assert len(generations) == len(messages) * num_return_sequences, \
        f"Expected {len(messages) * num_return_sequences} generations, but got {len(generations)}"
        
    return generations