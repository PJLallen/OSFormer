# -*- coding: utf-8 -*-

from collections import OrderedDict


def csv2tex(csv_path, out_path=None):
    if out_path is None:
        out_path = csv_path.replace('csv', '.txt')
    res = OrderedDict()
    res_og = OrderedDict()
    with open(csv_path, 'r') as f:
        flag = False
        for idx, line in enumerate(f):
            data_list = line.replace('\n', '').split(',')
            for ydx, elem in enumerate(data_list):
                if data_list[ydx] != '':
                    try:
                        if 'Max' in data_list[ydx]:
                            flt = float(data_list[ydx].replace('Max', ''))
                            data_list[ydx] = '\\textbf{' + '{:>4.1f}'.format(flt) + '}'
                        else:
                            flt = float(data_list[ydx])
                            data_list[ydx] = '{:>4.1f}'.format(flt)
                    except Exception as e:
                        pass
            if data_list[0] == '':
                if flag is False:
                    flag = True
                    continue
                break
            flag = False
            res[data_list[0]] = ' & '.join(data_list[1:])
            res_og[data_list[0]] = data_list[1:]

        with open(out_path, 'w') as f:
            for k, v in res.items():
                f.write('{:<15} & '.format(k) + v + ' \\\\ \n')


if __name__ == '__main__':
    csv2tex('OSFormer-NoVal-Attribute.csv')

