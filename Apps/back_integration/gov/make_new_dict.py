import json

from Apps.back_integration.gov.slot_config import requirement_weight

requirement_all = []
for i in range(len(requirement_weight)):
    requirement_all.append([])
for i in range(len(requirement_weight)):
    requirement_all[i] = list(requirement_weight[i].keys())
dict_list = set()


def make_dict(requirement):
    requirement.remove(requirement[0])
    requirement.remove(requirement[0])
    for _r in requirement:
        dict_list.add(_r.strip() + '\n')


for i in range(len(requirement_all)):
    make_dict(requirement_all[i])

with open('../data/new_dict.txt', 'w') as f:
    f.writelines(list(dict_list))


def generate_slot_set():
    dic = {}

    for i in range(len(requirement_weight)):
        dic = {**dic, **requirement_weight[i]}
    new_dict = list(dic.keys())
    slot_set = {}
    keys = new_dict
    v = 0
    for i in keys:
        slot_set[i] = v
        v += 1
    slot_set['service'] = v

    with open('../data/slot_set.json', 'w') as f:
        json.dump(slot_set, f, indent=4, ensure_ascii=False)


def generate_slot_max_weight():
    slot_max_weight = dict()
    slot_max = []
    for i in range(len(requirement_weight)):
        name = list(requirement_weight[i].keys())[1]
        slot_max_weight[name] = 100
        slot_max.append(name)
    # for i in range(len(requirement_weight)):
    #     for key, item in requirement_weight[i].items():
    #         if 40 < item < 100:
    #             slot_max.append(key)
    #             slot_max_weight[key] = 100
    with open('../data/slot_max_weight.json', 'w') as f:
        json.dump(slot_max_weight, f, indent=4, ensure_ascii=False)
    return slot_max


generate_slot_set()
generate_slot_max_weight()
