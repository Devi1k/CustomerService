import json
import os
import time

import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import onnxruntime as ort
import torch
from transformers import BertTokenizer, AutoTokenizer, AlbertForSequenceClassification

from Apps.ai_multiround.utils.logger import Logger, clean_log

log = Logger('multiround').getLogger()
tokenizer = AutoTokenizer.from_pretrained("voidful/albert_chinese_base")
mapper_list = ["true", "false"]
model_name = "multiround.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sess = ort.InferenceSession(os.getcwd() + "/Apps/ai_multiround/model/multiround.onnx",
                            providers=['CUDAExecutionProvider'])
model = AlbertForSequenceClassification.from_pretrained("voidful/albert_chinese_base")
model.load_state_dict(torch.load(os.getcwd() + "/Apps/ai_multiround/model/multiround.bin", map_location={0:device}))
model.eval()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def infer_torch(sentence1, sentence2):
    inputs = tokenizer.encode_plus(sentence1, sentence2, add_special_tokens=True, max_length=512, truncation=True)
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)
    input_len = len(input_ids)
    padding_length = 512 - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    input_ids = torch.LongTensor(input_ids).unsqueeze(0)
    attention_mask = torch.LongTensor(attention_mask).unsqueeze(0)
    token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0)
    labels = torch.LongTensor([0])
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels
    }
    outputs = model(**inputs)
    _, logits = outputs[:2]
    preds = logits.detach().cpu().numpy()
    predict_label = np.argmax(preds, axis=1)

    return mapper_list[predict_label[0]]


def infer_onnx(sess, utterance1, utterance2):
    inputs = tokenizer.encode_plus(utterance1, utterance2, add_special_tokens=True, max_length=512, truncation=True)
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)
    input_len = len(input_ids)
    padding_length = 512 - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    input_ids = torch.LongTensor(input_ids).unsqueeze(0)
    attention_mask = torch.LongTensor(attention_mask).unsqueeze(0)
    token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0)
    labels = torch.LongTensor([0])
    inputs = {
        "input_ids": to_numpy(input_ids),
        "attention_mask": to_numpy(attention_mask),
        "token_type_ids": to_numpy(token_type_ids),
        "labels": to_numpy(labels)
    }

    # input = {
    #     'word_seq_tensor': to_numpy(word_seq_tensor),
    #     'word_mask_tensor': to_numpy(word_mask_tensor),
    # }

    def get_output_name(onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    # input_name = sess.get_inputs()[0].name
    out_name = get_output_name(sess)
    output = sess.run(out_name, inputs)
    res = mapper_list[np.argmax(output)]
    print(res)
    return res


sentence1 = "我想办理公共卫生许可"
sentence2 = "我想挂个牌匾"
warm_start_time = time.time()
multi = infer_torch(sentence1, sentence2)
# multi = infer_onnx(sess,sentence1, sentence2)
log.info(
    'warm start text:{},{},intent:{},cost time:{}'.format(sentence1, sentence2, multi,
                                                          str(time.time() - warm_start_time)))


@csrf_exempt
def identify(request):
    log.info('*' * 5 + 'clean log' + '*' * 5)
    clean_log()
    log.info('-----------------------------------------------------------')
    if request.method == 'GET':
        pre_text = request.GET.get('pre_text')
        cur_text = request.GET.get('cur_text')
        if pre_text is None or cur_text is None:
            return JsonResponse({'message': 'params error',
                                 'code': 50012})
        start_time = time.time()
        # res = infer_onnx(sess, pre_text, cur_text)
        res = infer_torch(pre_text, cur_text)
        log.info('text:{},{},intent:{},cost time:{}'.format(pre_text, cur_text, res, str(time.time() - start_time)))
        return JsonResponse({'message': 'success', 'data': res, 'code': 0})
    return JsonResponse({'message': 'unknown methods',
                         'code': 50012})
