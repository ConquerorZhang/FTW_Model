# -*- coding: utf-8 -*-

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
#我的子函数
from Filter_15sui import Filter_15sui
from UserGouZaoTrain import UserGouZaoTrain
from User_X_Y_Train_or_Test import X_Y_Train_or_Test
from User_Save_Feature_Imp import User_Save_Feature_Imp
from User_Score_Offline import *
from User_Predict_Online import *

def User_model():
    '训练User模型，预测online有可能下单的User'
    #加载user特征的数据
    Biaotou_Train = pd.read_csv('../DataOutPut/user_features/train/Sample1_404_408_biaotou.csv',sep=',')#没有表头,训练数据的
    Sample1 = pd.read_csv('../DataOutPut/user_features/train/Sample1_404_408.csv', header=None,names=list(Biaotou_Train),sep=',')
    Sample1 = Filter_15sui(Sample1)#过滤15suiyixia并排索引
    Sample1_XiaDan = pd.read_csv('../DataOutPut/user_features/train/Sample1_409_413_XiaDan.csv',sep=',')

    Sample2 = pd.read_csv('../DataOutPut/user_features/train/Sample2_330_403.csv', header=None,names=list(Biaotou_Train),sep=',')
    Sample2 = Filter_15sui(Sample2)#过滤15suiyixia并排索引

    Biaotou_Test = pd.read_csv('../DataOutPut/user_features/Online_test/test_predict_416_420_biaotou.csv',sep=',')#没有表头,Online数据的
    Online_predict = pd.read_csv('../DataOutPut/user_features/Online_test/test_predict_416_420.csv',header=None,names=list(Biaotou_Test),sep=',')
    Online_predict = Filter_15sui(Online_predict)#过滤15suiyixia并排索引

    Offline_train_Temp = Sample2 #训练集数据
    Offline_test_Sample1_Temp = Sample1  # 线下测试数据，方便转换sample_i
    Offline_XiaDan_Sample1_Temp = Sample1_XiaDan  # 线下测试下单的数据，方便转换sample_i

    # 线下测试集数据的X和Y
    X_Sample1_test, Y_Sample1_test = X_Y_Train_or_Test(Offline_test_Sample1_Temp)

    #负样本采样
    train_Neg_frac = 0.08
    #Frac = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12] #比较线下结果，选比较好的负样本抽样系数
    #for train_Neg_frac in Frac:
    while True:
        train = UserGouZaoTrain(Offline_train_Temp,train_Neg_frac) #构造训练样本数据（负样本有随机抽样）

        # 保存较好的训练样本
        train.to_csv('./user_CSV/User_Sample_train.csv',index=False)

        # train是否使用之前保存的效果较好的样本
        #train = pd.read_csv('./user_CSV/User_Sample_train.csv', sep=',')

        # 训练数据的X和Y
        X_train, Y_train = X_Y_Train_or_Test(train)

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
                                            seed=10,nthread=3,objective="binary")
        '''
        # 建立模型
        gbm = lgb.LGBMClassifier(num_leaves=8, max_depth=2, n_estimators=460, learning_rate=0.01, subsample=0.8,
                                            seed=10,nthread=3,objective="binary")
        gbm.fit(X_train, Y_train)

        # 保存feature_importances_
        #Save_Feature_Imp(X_train, gbm.feature_importances_)

        #线下测试集预测：
        y_pre_S1 = gbm.predict(X_Sample1_test)

        #线下测试集评分：
        print '..........................User Model :'
        F11_Sample1,Sample1_User, userIdCorrectNum1, lenPred,F11Precise,F11Recall = User_Score_Offline(y_pre_S1, gbm, X_Sample1_test, Offline_test_Sample1_Temp, Offline_XiaDan_Sample1_Temp)
        #Sample1_User.to_csv('./user_CSV/Sample1_User.csv',index=False)

        #线上测试集预测：
        lenPreOnlineData = User_Predict_Online(Online_predict,gbm)

        #固定比较好的训练样本的截止条件
        if F11Precise > 0.17 and F11Recall > 0.21: #and userIdCorrectNum1>250: # and lenPred<1501:
            if lenPreOnlineData<901 and lenPreOnlineData>799:
                break
