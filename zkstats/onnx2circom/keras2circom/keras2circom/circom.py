# Ref: https://github.com/zk-ml/uchikoma/blob/main/python/uchikoma/circom.py

import typing

import os
from os import path
from dataclasses import dataclass

import re


@dataclass
class Template:
    op_name: str
    fpath: str

    args: typing.List[str]

    input_names: typing.List[str]   = None
    input_dims: typing.List[int]    = None
    output_names: typing.List[str]  = None
    output_dims: typing.List[int]   = None

    def __str__(self) -> str:
        args_str = ', '.join(self.args)
        args_str = '(' + args_str + ')'
        return '{:>20}{:30} {}{}{}{} \t<-- {}'.format(
            self.op_name, args_str,
            self.input_names, self.input_dims,
            self.output_names, self.output_dims,
            self.fpath)

def file_parse(fpath):
    '''parse circom file and register templates'''
    with open(fpath, 'r') as f:
        lines = f.read().split('\n')

    lines = [l for l in lines if not l.strip().startswith('//')]
    lines = ' '.join(lines)

    lines = re.sub('/\*.*?\*/', 'IGN', lines)

    # !@# file_parse: op_name='TFReduceSum'
    # !@# file_parse: signals=[('input', 'in', '[nInputs][1]'), ('output', 'out', '[1]')]
    # !@# file_parse: sig=('input', 'in', '[nInputs][1]')
    # !@# file_parse: sig=('output', 'out', '[1]')
    # !@# file_parse: infos=[['in'], [2], ['out'], [1]]
    # !@# file_parse: args=['nInputs']
    templates: typing.Dict[str, Template] = {}
    funcs = re.findall('template (\w+) ?\((.*?)\) ?\{(.*?)\}', lines)
    for func in funcs:
        op_name = func[0].strip()
        args_str = func[1]
        # If `args_str` is empty, it means there is no arg and `args` will be an empty list
        if args_str == '':
            args = []
        else:
            args = func[1].split(',')
        main = func[2].strip()
        assert op_name not in templates, \
            'duplicated template: {} in {} vs. {}'.format(
                    op_name, templates[op_name].fpath, fpath)
        print(f"!@# file_parse: {op_name=}")
        signals = re.findall('signal (\w+) (\w+)(.*?);', main)
        print(f"!@# file_parse: {signals=}")
        infos = [[] for i in range(4)]
        # E.g. sig = ('input', 'in', '[nInputs][1]')
        for sig in signals:
            print(f"!@# file_parse: {sig=}")
            sig_types = ['input', 'output']
            assert sig[0] in sig_types, sig[1] + ' | ' + main
            idx = sig_types.index(sig[0])
            # infos[0] contains the names of the signals
            # idx = 0 ->
            #   - infos[0]: input signal names
            #   - infos[1]: input signal dims (number of [])
            # idx = 1 ->
            #   - infos[2]: output signal names
            #   - infos[3]: output signal dims (number of [])
            infos[idx*2+0].append(sig[1])

            sig_dim = sig[2].count('[')
            infos[idx*2+1].append(sig_dim)
        templates[op_name] = Template(
            op_name=op_name,
            fpath=fpath,
            args=[a.strip() for a in args],
            input_names=infos[0],
            input_dims=infos[1],
            # input_shape
            output_names=infos[2],
            output_dims=infos[3],
        )
    return templates


def dir_parse(dir_path, skips=[]):
    '''parse circom files in a directory'''
    names = os.listdir(dir_path)
    for name in names:
        if name in skips:
            continue

        fpath = path.join(dir_path, name)
        if os.path.isdir(fpath):
            dir_parse(fpath)
        elif os.path.isfile(fpath):
            if fpath.endswith('.circom'):
                file_parse(fpath)
