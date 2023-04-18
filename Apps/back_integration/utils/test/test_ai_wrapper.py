from unittest import TestCase

import xlrd

from ..ai_wrapper import *


class Test(TestCase):

    def test_get_related_title(self):
        sentence = "个人手机业务"
        res = get_related_title(sentence)
        print(res)

    def test_get_faq_from_service(self):
        # score, answer, service = get_faq_from_service("需要第三方吗",
        #                                               "固定资产投资项目合理用能审查")  # add assertion here
        sentence = "我要办理机动车维修经营备案，需要什么条件"

        service = "机动车维修经营备案"

        score, answer, service = get_faq_from_service(sentence,
                                                      service, history=[])  # add assertion here

        print(score, answer)

    def test_get_recommend(self):
        service_name = "户外广告及临时悬挂、设置标语或者宣传品许可-户外广告设施许可（不含公交候车亭附属广告及公交车体广告设施）（市级权限委托市内六区实施）"
        history = ['我想挂个牌匾']
        res = get_recommend(service_name, history)
        print(res)

    def test_get_faq(self):
        with open("./data/link.json", "r", encoding="utf-8") as f:
            link = json.load(f)
        total = len(link)
        count = 0
        for title in link.keys():
            print(title)
            res = get_faq(title)
            if res == "您需要办理" + title:
                count += 1
        print(count / total)

        # read xlsx
        book = xlrd.open_workbook("./data/20230407测试集.xlsx")
        sheet = book.sheet_by_index(2)
        total = sheet.nrows - 1
        count = 0
        for i in range(1, total + 1):
            query = sheet.cell_value(i, 0)
            service = sheet.cell_value(i, 1)
            answer = sheet.cell_value(i, 2)
            similarity, res, service = get_faq(query)
            if similarity > 0.9:
                if res == answer:
                    count += 1
            else:
                similarity, res, service = get_faq_from_service(query, service, [])
                if similarity > 0.4:
                    if res == answer:
                        count += 1
        print(count / total)

    def test_multi_round(self):

        # read xlsx
        book = xlrd.open_workbook("./data/20230407测试集.xlsx")
        sheet = book.sheet_by_index(1)
        total = sheet.nrows - 1
        count = 0
        for i in range(1, total + 1):
            query1 = sheet.cell_value(i, 0)
            query2 = sheet.cell_value(i, 1)
            label = sheet.cell_value(i, 2)
            res = is_multi_round(query1, query2)
            if res == label:
                count += 1
        print("multi round test result is {}".format(round(count / total, 2)))

    def test_matter_content(self):

        # read xlsx
        book = xlrd.open_workbook("./data/20230407测试集.xlsx")
        sheet = book.sheet_by_index(0)
        total = sheet.nrows - 1
        count = 0
        for i in range(1, total + 1):
            query = sheet.cell_value(i, 0)
            label = sheet.cell_value(i, 1)
            res = get_item_content(query)
            if res == label:
                count += 1
        print("matter content test result is {}".format(round(count / total, 2)))

    def test_business(self):
        # read xlsx
        book = xlrd.open_workbook("./data/20230407测试集.xlsx")
        sheet = book.sheet_by_index(3)
        total = sheet.nrows - 1
        count = 0
        for i in range(1, total + 1):
            query = sheet.cell_value(i, 0)
            label = sheet.cell_value(i, 1)
            res = get_business(query)
            if res == label:
                count += 1
        print("business test result is {}".format(round(count / total, 2)))

