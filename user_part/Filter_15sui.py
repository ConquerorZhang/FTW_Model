# -*- coding: utf-8 -*-

def Filter_15sui(Sample):
    "过滤15suiyixia的user"
    Sample = Sample[~(Sample.age == '15suiyixia')]  # 过滤15岁以下的
    Sample = Sample.reset_index(drop=True)  # 重新排索引
    return Sample