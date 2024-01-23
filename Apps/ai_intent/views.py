import json
import os
import time
from time import strftime, gmtime

import numpy as np
import onnxruntime as ort
from django.http import JsonResponse
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from .nlp_cls.dataloader import Dataloader
from .util.logger import Logger

log = Logger('intent').getLogger()


def build_predict_text_raw(text, context=None):
    if context is None:
        context = list()
    device = "cpu"
    # tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    ori_word_seq = dataloader.tokenizer.tokenize(text)
    # ori_tag_seq = ['O'] * len(ori_word_seq)
    context_size = 1
    context_seq = dataloader.tokenizer.encode('[CLS] ' + ' [SEP] '.join(context[-context_size:]))
    intents = []
    multi_round = []
    da = {}

    word_seq, new2ori = ori_word_seq, None
    batch_data = [[ori_word_seq, intents, [], [], context_seq,
                   new2ori, word_seq, dataloader.seq_intent2id(intents)]]

    pad_batch = dataloader.pad_batch(batch_data)
    pad_batch = tuple(t.to(device) for t in pad_batch)
    # word_seq_tensor, intent_tensor, word_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
    # context_seq_tensor, context_mask_tensor = None, None
    return pad_batch


def build_predict_text(text):
    pad_batch = build_predict_text_raw(text)
    # pad_batch = tuple(t.to("gpu:0") for t in pad_batch)
    word_seq_tensor, intent_tensor, word_mask_tensor, word_token_type_ids, context_seq_tensor, context_mask_tensor = pad_batch
    intent_tensor = None
    context_seq_tensor, context_mask_tensor = None, None
    return word_seq_tensor, intent_tensor, word_mask_tensor, word_token_type_ids, context_seq_tensor, context_mask_tensor


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def infer_onnx(sess, utterance):
    word_seq_tensor, intent_tensor, word_mask_tensor, word_token_type_ids, context_seq_tensor, context_mask_tensor = build_predict_text(
        utterance)
    model_input = {
        'input_ids': to_numpy(word_seq_tensor),
        'attention_mask': to_numpy(word_mask_tensor),
        'token_type_ids': to_numpy(word_token_type_ids)
    }

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
    intent = sess.run(out_name, model_input)
    intent_index = np.argmax(intent)
    return dataloader.id2intent[intent_index]


log.info('model loading')
project_path = os.getcwd() + "/Apps/ai_intent/"
intent_vocab = json.load(open(project_path + 'processed_data/intent_vocab.json'))

model_path = "/home/l/CustomerService/chinese-bert-wwm-ext"
dataloader = Dataloader(intent_vocab=intent_vocab,
                        pretrained_weights=model_path)
log.info(ort.__version__)
log.info(ort.get_device())
sess = ort.InferenceSession(project_path + 'output/ai_intent.onnx', providers=['CUDAExecutionProvider'])

log.info('warming up')

start_time = time.time()
log.info('text:人才绿卡A卡的办理条件,intent:{},cost_time:{}'.format(infer_onnx(sess,
                                                                               '人才绿卡A卡的办理条件'),
                                                                    str(time.time() - start_time)))
log.info('warm up finish. Waiting for messages')


def clean_log():
    path = os.getcwd() + '/Apps/ai_intent/log/'
    for i in os.listdir(path):
        if len(i) < 7:
            continue
        file_path = path + i  # 生成日志文件的路径
        timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        # 获取日志的年月，和今天的年月
        today_m = int(timestamp[4:6])  # 今天的月份
        file_m = int(i[12:14])  # 日志的月份
        today_y = int(timestamp[0:4])  # 今天的年份
        file_y = int(i[7:11])  # 日志的年份
        # 对上个月的日志进行清理，即删除。
        # print(file_path)
        if file_m < today_m:
            if os.path.exists(file_path):  # 判断生成的路径对不对，防止报错
                os.remove(file_path)  # 删除文件
        elif file_y < today_y:
            if os.path.exists(file_path):
                os.remove(file_path)


@csrf_exempt
def intent_cls(request):
    # print(request.method)
    log.info('*' * 5 + 'clean log' + '*' * 5)
    clean_log()
    log.info('-----------------------------------------------------------')
    if request.method == 'GET':
        raw_text = request.GET.get('text').strip()
        if raw_text == "":
            return JsonResponse({'message': 'message format error',
                                 'code': 50012})
        intent = infer_onnx(sess, raw_text)
        log.info('text:{},intent:{}'.format(raw_text, intent))
        return JsonResponse({'message': 'success', 'data': intent, 'code': 0})
    return JsonResponse({'message': 'unknown methods',
                         'code': 50012})
