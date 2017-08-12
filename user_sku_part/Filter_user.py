# -*- coding: utf-8 -*-

import pandas as pd
def Filter_user(Sample,User):
    "在user-sku模型的线上user数据中，选择user模型选出来的user"
    Sample = pd.merge(Sample, User)
    Sample = Sample.reset_index(drop=True)  # 重新排索引
    return Sample