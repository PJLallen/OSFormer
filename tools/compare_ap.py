import os
import json
import numpy as np
from collections import defaultdict, OrderedDict


basedir = 'SOTA'
base_method = 'osformer'
datasets = ['cod', 'nc4k']
file_template = 'my_data_test_coco_{}_style_ap.json'


for dataset in datasets:

    with open(os.path.join(basedir, base_method, file_template.format(dataset)), 'r') as f:
        ours = json.load(f)
        print(base_method, len(ours.keys()))

    delta_dict = defaultdict(list)

    for method in os.listdir(basedir):
        if method == base_method or not os.path.isdir(os.path.join(basedir, method)):
            continue

        with open(os.path.join(basedir, method, file_template.format(dataset)), 'r') as f:
            other = json.load(f)
            print(method, len(other.keys()))

        for k, v in ours.items():
            our_ap = float(ours[k]['AP'])
            other_ap = float(other[k]['AP']) if other.get(k) else 0
            if np.isnan(other_ap):
                other_ap = 0
            delta_dict[k].append(our_ap - other_ap)

    od = []
    for k, v in delta_dict.items():
        print(k, v)
        od.append((k, np.mean(v)))

    od.sort(key=lambda x: x[1], reverse=True)
    res = OrderedDict()
    for elem in od:
        res[elem[0]] = elem[1]

    with open(os.path.join(basedir, 'desc_res_{}.json').format(dataset), 'w') as f:
        json.dump(res, f, indent=4)

