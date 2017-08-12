# -*- coding: utf-8 -*-

import pandas as pd

def User_Save_Feature_Imp(X_train,Feature_Imp):
    "保存feature_importances_"
    Feature_Imp = pd.Series(Feature_Imp)
    Feature_Names = pd.Series(X_train.columns)  # 特征的列名称
    Feature_Imp = pd.concat([Feature_Names, Feature_Imp], axis=1)
    Feature_Imp.to_csv('./user_CSV/User_Feature_importances.csv', header=False, index=False)  # 不带索引