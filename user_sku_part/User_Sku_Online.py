# -*- coding: utf-8 -*-

import lightgbm as lgb
#我的子函数
from Filter_15sui import Filter_15sui
from Filter_user import Filter_user
from X_Y_Train_or_Test import X_Y_Train_or_Test
from Predict_Online import *

def User_Sku_Online():
    '预测online会下单User的Sku'
    # 加载online数据
    Biaotou_Test = pd.read_csv('../DataOutPut/user_sku_features/Online_test/test_predict_416_420_biaotou.csv',sep=',')  # 没有表头,Online数据的
    Online_predict = pd.read_csv('../DataOutPut/user_sku_features/Online_test/test_predict_416_420.csv',header=None, names=list(Biaotou_Test), sep=',')
    Online_predict = Filter_15sui(Online_predict)  # 过滤15suiyixia并排索引
    Online_User = pd.read_csv('./user_CSV/Online_User.csv', sep=',') #加载User模型生成的User
    Online_predict = Filter_user(Online_predict,Online_User)#过滤user

    # train使用之前保存的效果较好的样本
    train = pd.read_csv('./user_sku_CSV/Sample_train.csv', sep=',')

    # 训练数据的X和Y
    X_train,Y_train = X_Y_Train_or_Test(train)

    # 建立模型
    gbm = lgb.LGBMClassifier(num_leaves=8, max_depth=3, n_estimators=220, learning_rate=0.05, subsample=0.8,#max_bin=750,
                             seed=10,nthread=3,objective="binary")

    gbm.fit(X_train,Y_train)

    # 线上预测：
    Predict_Online_UserModel(Online_predict,gbm)
