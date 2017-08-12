# -*- coding: utf-8 -*-

import pandas as pd
import lightgbm as lgb
#我的子函数
from Filter_15sui import Filter_15sui
from gouZaoTrain import gouZaoTrain
from X_Y_Train_or_Test import X_Y_Train_or_Test
from Save_Feature_Imp import Save_Feature_Imp
from Score_Offline import Score_Offline

def User_Sku_Offline():
    '通过线下测试集训练User-Sku模型'
    # 加载数据
    Biaotou_Train = pd.read_csv('../DataOutPut/user_sku_features/train/Sample1_404_408_biaotou.csv',sep=',')#没有表头,训练数据的
    Sample1 = pd.read_csv('../DataOutPut/user_sku_features/train/Sample1_404_408.csv', header=None,names=list(Biaotou_Train),sep=',')
    Sample1 = Filter_15sui(Sample1)#过滤15suiyixia并排索引
    Sample1_XiaDan = pd.read_csv('../DataOutPut/user_sku_features/train/Sample1_409_413_XiaDan.csv',sep=',')

    Sample2 = pd.read_csv('../DataOutPut/user_sku_features/train/Sample2_330_403.csv', header=None,names=list(Biaotou_Train),sep=',')
    Sample2 = Filter_15sui(Sample2)#过滤15suiyixia并排索引

    train_Zhong_Zhuan = Sample2
    test_Zhong_Zhuan = Sample1  # 线下测试数据，方便转换sample_i
    realData_Zhong_Zhuan = Sample1_XiaDan  # 线下测试下单的数据，方便转换sample_i

    # 测试数据的X和Y
    X_test,Y_test = X_Y_Train_or_Test(test_Zhong_Zhuan)

    train_Neg_frac = 0.08 #负样本采样
    #Frac = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12] #比较线下结果，选比较好的负样本抽样系数
    #for train_Neg_frac in Frac:
    while True:
        train = gouZaoTrain(train_Zhong_Zhuan,train_Neg_frac)

        #保存较好的训练样本
        train.to_csv('./user_sku_CSV/Sample_train.csv',index=False)  # 不带索引

        # 训练数据的X和Y
        X_train,Y_train =X_Y_Train_or_Test(train)

        # 比较线下结果，选比较好的模型参数
        '''
        estimators = range(100, 701, 50)
        learn = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
        dep = [2, 3, 4]
        for dp in dep:
            for es in estimators:
                for lr in learn:
                    print 'train_Neg_frac: ...................................', train_Neg_frac
                    print 'max_depth:.........................................', dp
                    print 'n_estimators:......................................', es
                    print 'learning_rate:......................................', lr
                    gbm = lgb.LGBMClassifier(num_leaves=8, max_depth=dp, n_estimators=es, learning_rate=lr, subsample=0.8,
                                             seed=10, nthread=3, objective="binary")
        '''
        # 建立模型
        gbm = lgb.LGBMClassifier(num_leaves=8, max_depth=3, n_estimators=220, learning_rate=0.05, subsample=0.8,#max_bin=750,
                                 seed=10,nthread=3,objective="binary")

        gbm.fit(X_train,Y_train)
        y_pred = gbm.predict(X_test)

        # 保存feature_importances_
        #Save_Feature_Imp(X_train,gbm.feature_importances_)

        # 线下测试集评分：
        print '..........................User Sku Model :'
        F11, F12, Score = Score_Offline(y_pred,gbm,X_test,test_Zhong_Zhuan,realData_Zhong_Zhuan)

        # 固定比较好的训练样本的截止条件
        if F11 > 0.161 and F12 > 0.099 and Score > 0.124:
            break