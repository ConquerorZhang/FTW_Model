# -*- coding: utf-8 -*-

import pandas as pd

def X_Y_Train_or_Test(Data):
    "训练or测试数据的X和Y"
    x_columns = [x for x in Data.columns if x not in ['user_sku_id', 'user_id', 'sku_id', 'y']]
    Y_train = Data['y']
    X_train_tmp = Data[x_columns]  # 去掉user_sku_id和y列的
    X_train = pd.get_dummies(X_train_tmp, prefix=['age', 'sex', 'attr1', 'attr2', 'attr3'],
                             columns=['age', 'sex', 'attr1', 'attr2', 'attr3'])
    return X_train,Y_train