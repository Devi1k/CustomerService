import json

from ..gov.slot_config import requirement_weight

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


generate_slot_set()
