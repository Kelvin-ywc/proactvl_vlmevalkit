import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import copy as cp
from .base import BaseModel
from ..smp import isimg, listinstr
from ..dataset import DATASET_TYPE


class ProactVL(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self):
        assert model_path is not None
        self.model_path = model_path
        self.model_path = 'oaaoaa/proact_all_fulltuning_base_qwen3vl_final'
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
        default_kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=1,
            num_return_sequences=1,
            use_cache=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def adjust_kwargs(self, dataset):
        kwargs = cp.deepcopy(self.kwargs)
        if DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
            kwargs['max_new_tokens'] = 32
        elif DATASET_TYPE(dataset) == 'Caption' and 'COCO' in dataset:
            kwargs['max_new_tokens'] = 32
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['OCRVQA', 'ChartQA', 'DocVQA'], dataset):
                kwargs['max_new_tokens'] = 100
            elif listinstr(['TextVQA'], dataset):
                kwargs['max_new_tokens'] = 10
        return kwargs

    def generate_inner(self, message, dataset=None):
        if dataset is not None:
            kwargs = self.adjust_kwargs(dataset)
        else:
            kwargs = self.kwargs
        prompt = ''
        for s in message:
            if s['type'] == 'image':
                prompt += f'<img>{s["value"]}</img>'
            elif s['type'] == 'text':
                prompt += s['value']
        if dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            prompt += ' Answer:'
        encoded = self.tokenizer([prompt], return_tensors='pt', padding='longest')
        input_ids = encoded.input_ids.to('cuda')
        attention_mask = encoded.attention_mask.to('cuda')

        pred = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs)
        answer = self.tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        return answer

from src.infer.multi_assistant_inference import MultiAssistantStreamInference
class ProactVLChat(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self):
        self.model_path = 'oaaoaa/proact_all_fulltuning_base_qwen3vl_final'
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map='cuda', trust_remote_code=True).eval()
        # ckpt_path = args.ckpt_path
        weight_dir_prefix = ''

        infer_config = {
            'use_audio_in_video': False,
            'max_kv_tokens': 16384,
            'assistant_num': 1,
            'enable_tts': False,
            'save_dir': './infer_output',
        }
        generate_config = {
            'do_sample': True,
            'max_new_tokens': 12,
            'temperature': 0.7,
            'top_p': 0.9,
            'repetition_penalty': 1.15,
        }

        self.infer = MultiAssistantStreamInference(None, self.model_path, weight_dir_prefix, infer_config, generate_config, None, 'cuda')
        torch.cuda.empty_cache()

    def build_history(self, message):

        def concat_tilist(tilist):
            image_cnt = 1
            prompt = ''
            for item in tilist:
                if item['type'] == 'text':
                    prompt += item['value']
                elif item['type'] == 'image':
                    prompt += f"Picture {image_cnt}: <img>{item['value']}</img>\n"
                    image_cnt += 1
            return prompt

        assert len(message) % 2 == 0
        hist = []
        for i in range(len(message) // 2):
            m1, m2 = message[2 * i], message[2 * i + 1]
            assert m1['role'] == 'user' and m2['role'] == 'assistant'
            hist.append((concat_tilist(m1['content']), concat_tilist(m2['content'])))
        return hist

    def generate_inner(self, message, dataset=None):
        self.infer.assistants[0].clear_session()
        assert message[0]['role'] == 'system'
        # self.infer.assistants[0].prime_system_prompt(message[0]['value'])
        # message = [s for s in message if ('role' not in s or 'role' in s and s['role'] != 'system')]
        vl_list = [{'image': s['value']} if s['type'] == 'image' else {'text': s['value']} for s in message]
        # query = self.infer.tokenizer.from_list_format(vl_list)
        conversation = [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': message[0]['value']
                    }
                ]
            },
            {
                'role': 'user',
                'content': [

                ]
            }
        ]
        # conversation = []
        for item in vl_list[1:]:
            if 'text' in item:
                conversation[1]['content'].append({"type": "text", "text": item['text']})
            elif 'image' in item:
                conversation[1]['content'].append({"type": "image", "image": item['image']})

        print(conversation)
        inputs = self.infer.model.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.infer.model.device).to(self.infer.model.llm.dtype)
        # print(inputs.keys())
        self.infer.model.config.num_hidden_layers = 36
        self.infer.assistants[0].forward_cache(inputs, generate_flag=False)
        response = self.infer.assistants[0].forward_assistant()
        # print(generated_ids)
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        # print(generated_ids_trimmed)   
        # output_text = self.infer.model.processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )
        # print(output_text)    
        # exit(0) 
        # print(query)
        # exit(0)
        # response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        return response

    def chat_inner(self, message, dataset=None):
        assert len(message) % 2 == 1 and message[-1]['role'] == 'user'
        history = self.build_history(message[:-1])
        vl_list = [
            {'image': s['value']} if s['type'] == 'image' else {'text': s['value']}
            for s in message[-1]['content']
        ]
        query = self.tokenizer.from_list_format(vl_list)
        response, _ = self.model.chat(self.tokenizer, query=query, history=history, **self.kwargs)
        return response
