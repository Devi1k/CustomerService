import logging
import os.path
from unittest import TestCase

import xlrd
import xlwt
from tqdm import tqdm

from ..ai_wrapper import *
from ..word_match import longestCommonSubsequence, lev, sigmoid

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - File \"%(name)s/%(filename)s\" - line %(lineno)s - %(levelname)s - %(message)s')
log = logging.getLogger()


class Test(TestCase):

    def test_get_related_title(self):
        sentence = "我要开豆腐摊，要怎么办理？"
        res = get_related_title(sentence)
        log.info(res)

    def test_get_faq_from_service(self):
        sentence = "我要办理设置医疗机构许可和执业登记，是一次办事项吗"

        service = "设置医疗机构许可和执业登记、变更-设置医疗机构许可-设置内资医疗机构许可"

        score, answer, service = get_faq_from_service(sentence,
                                                      service, log=log)  # add assertion here

        log.info("score{},answer{}".format(score, answer))

    def test_get_recommend(self):
        service_name = "户外广告及临时悬挂、设置标语或者宣传品许可-户外广告设施许可（不含公交候车亭附属广告及公交车体广告设施）（市级权限委托市内六区实施）"
        history = ['我想挂个牌匾']
        res = get_recommend(service_name, history)
        log.info(res)

    def test_get_faq(self):

        # read xlsx
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/20230504测试集.xlsx")
        # read xlsx
        book = xlrd.open_workbook(file_path)
        sheet = book.sheet_by_index(1)
        total = sheet.nrows - 1
        count = 0
        # export to xlsx
        output_workbook = xlwt.Workbook(encoding="utf-8")
        output_worksheet = output_workbook.add_sheet("sheet1")
        output_row = 1
        output_worksheet.write(0, 0, "query")
        output_worksheet.write(0, 1, "service_label")
        output_worksheet.write(0, 2, "res")
        output_worksheet.write(0, 3, "answer_label")
        output_worksheet.write(0, 4, "similarity")
        for i in tqdm(range(1, total + 1)):
            query = sheet.cell_value(i, 0).strip()
            service_label = sheet.cell_value(i, 1).strip().replace("--", "-")
            answer = str(sheet.cell_value(i, 2)).strip().replace("\\n", "\n").replace("\\t", "")
            # if len(answer) > 1:
            #     print(answer)
            try:
                score, res, service = get_faq(query)
            except Exception:
                log.info(query)
                log.info("connect error")
                score, res = 0, ""
            if score > 0.94:
                if Levenshtein.ratio(res, answer) > 0.98 or score == 1.0:
                    count += 1
                else:
                    output_worksheet.write(output_row, 0, query)
                    output_worksheet.write(output_row, 1, service_label)
                    output_worksheet.write(output_row, 2, res)
                    output_worksheet.write(output_row, 3, answer)
                    output_worksheet.write(output_row, 4, score)
                    output_row += 1
            else:
                similarity, res, service = get_faq_from_service(query, service_label, log=log)
                if similarity > 0.285:
                    if Levenshtein.ratio(res, answer) > 0.98:
                        count += 1
                    else:
                        output_worksheet.write(output_row, 0, query)
                        output_worksheet.write(output_row, 1, service_label)
                        output_worksheet.write(output_row, 2, res)
                        output_worksheet.write(output_row, 3, answer)
                        output_worksheet.write(output_row, 4, similarity)
                        output_row += 1

        output_workbook.save(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result/faq_test_result{}.xls"))
        log.info("faq test result is {}".format(round(count / total, 2)))
        log.info("error item {}".format(total - count))

    def test_multi_round(self):
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/20230407测试集.xlsx")
        # read xlsx
        book = xlrd.open_workbook(file_path)
        sheet = book.sheet_by_index(1)
        total = sheet.nrows - 1
        count = 0
        # export to xlsx
        output_workbook = xlwt.Workbook(encoding="utf-8")
        output_worksheet = output_workbook.add_sheet("sheet1")
        output_row = 1
        output_worksheet.write(0, 0, "query1")
        output_worksheet.write(0, 1, "query2")
        output_worksheet.write(0, 2, "label")
        output_worksheet.write(0, 3, "result")
        for i in tqdm(range(1, total + 1)):
            query1 = sheet.cell_value(i, 0).strip()
            query2 = sheet.cell_value(i, 1).strip()
            label = sheet.cell_value(i, 2)
            label = False if label == 0 else True
            res = is_multi_round(query1, query2)
            if res == label:
                count += 1
            else:
                output_worksheet.write(output_row, 0, query1)
                output_worksheet.write(output_row, 1, query2)
                output_worksheet.write(output_row, 2, label)
                output_worksheet.write(output_row, 3, res)
                output_row += 1

        output_workbook.save(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result/multi_round_test_result.xls"))
        log.info("multi round test result is {}".format(round(count / total, 2)))

    def test_matter_content(self):
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/20230407测试集.xlsx")
        # read xlsx
        book = xlrd.open_workbook(file_path)
        sheet = book.sheet_by_index(0)
        total = sheet.nrows - 1
        count = 0
        # export to xlsx
        output_workbook = xlwt.Workbook(encoding="utf-8")
        output_worksheet = output_workbook.add_sheet("sheet1")
        output_row = 1
        output_worksheet.write(0, 0, "query")
        output_worksheet.write(0, 1, "label")
        output_worksheet.write(0, 2, "result")
        for i in range(1, total + 1):
            query = sheet.cell_value(i, 0).strip()
            label = sheet.cell_value(i, 1).strip()
            res = get_item_content(query)
            if res == label:
                count += 1
            else:
                output_worksheet.write(output_row, 0, query)
                output_worksheet.write(output_row, 1, label)
                output_worksheet.write(output_row, 2, res)
                output_row += 1

        output_workbook.save(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result/matter_content_test_result.xls"))
        log.info("matter content test result is {}".format(round(count / total, 2)))

    def test_business(self):
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/20230407测试集.xlsx")
        # read xlsx
        book = xlrd.open_workbook(file_path)
        sheet = book.sheet_by_index(3)
        total = sheet.nrows - 1
        count = 0
        # export to xlsx
        output_workbook = xlwt.Workbook(encoding="utf-8")
        output_worksheet = output_workbook.add_sheet("sheet1")
        output_row = 1
        output_worksheet.write(0, 0, "query")
        output_worksheet.write(0, 1, "label")
        output_worksheet.write(0, 2, "result")
        for i in tqdm(range(1, total + 1)):
            query = sheet.cell_value(i, 0).strip()
            service_name = sheet.cell_value(i, 1).strip()
            label = sheet.cell_value(i, 2).strip()
            try:
                res = get_business(query, service_name)
            except:
                res = "网络错误"
            res = res if res != "无匹配业务" else "/"
            if res == label:
                count += 1
            else:
                output_worksheet.write(output_row, 0, query)
                output_worksheet.write(output_row, 1, label)
                output_worksheet.write(output_row, 2, res)
                output_row += 1
        output_workbook.save(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result/business_test_result.xls"))
        log.info("business test result is {}".format(round(count / total, 2)))

    def test_retrieval(self):
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/20230407测试集.xlsx")
        # read xlsx
        book = xlrd.open_workbook(file_path)
        sheet = book.sheet_by_index(4)
        total = sheet.nrows - 1
        count = 0
        # export to xlsx
        output_workbook = xlwt.Workbook(encoding="utf-8")
        output_worksheet = output_workbook.add_sheet("sheet1")
        output_row = 1
        output_worksheet.write(0, 0, "query")
        output_worksheet.write(0, 1, "label")
        output_worksheet.write(0, 2, "result")
        for i in tqdm(range(1, total + 1)):
            query = sheet.cell_value(i, 0).strip()
            service = sheet.cell_value(i, 1).strip().replace("--", "-")
            res = get_related_title(query)
            for index in range(len(res)):
                res[index] = res[index].replace("--", "-")
            if service in res:
                count += 1
            else:
                output_worksheet.write(output_row, 0, query)
                output_worksheet.write(output_row, 1, service)
                output_worksheet.write(output_row, 2, ",".join(res))
                output_row += 1

        output_workbook.save(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result/retrieval_test_result.xls"))
        log.info("retrieval test result is {}".format(round(count / total, 2)))

    def test_auto_bussiness(self):
        # query_list = ["残疾人家庭设施无障碍改造申请时间",""]
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/20230504测试集.xlsx")
        # read xlsx
        book = xlrd.open_workbook(file_path)
        sheet = book.sheet_by_index(4)
        total = sheet.nrows - 1
        count = 0
        max_score = 0
        for i in tqdm(range(1, total + 1)):
            query = sheet.cell_value(i, 0).strip()
            service = sheet.cell_value(i, 1).strip().replace("--", "-")
            # for query in query_list:
            options = get_related_title(query)
            # 若对话内容包含的事项足够明确
            business_threshold = 0.90
            candidate_service = ""
            for o in options:
                lcs = longestCommonSubsequence(query, o)
                if lcs <= 2:
                    continue
                distance = lev(query, o, True, True)
                distance = sigmoid(distance + lcs / len(o))
                if max_score < distance:
                    max_score = distance
                    candidate_service = o
            candidate_service = candidate_service.replace("--", "-")
            if max_score > business_threshold:
                if candidate_service == service:
                    count += 1
                else:
                    print("match failed label:{},result:{}".format(service, candidate_service))
            else:
                print("match failed")
        print(count / total)

if __name__ == "__main__":
    t1 = Test()
    # t1.test_get_faq_from_service()
    t1.test_business()
    t1.test_multi_round()
    t1.test_get_faq()
    t1.test_matter_content()
    t1.test_retrieval()
