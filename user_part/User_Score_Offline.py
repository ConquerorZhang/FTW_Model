# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def User_Score_Offline(y_pred, clf, X_test, test_Zhong_Zhuan, realData_Zhong_Zhuan):
    "线下评分，pro>0.5的user"
    y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
    y_pred_pro_tmp = clf.predict_proba(X_test)  # 算概率，去重用
    y_pred_pro = pd.DataFrame(y_pred_pro_tmp[:, 1], columns=['y_pred_pro'])  # 取y=1的概率
    
    New_test = pd.concat([test_Zhong_Zhuan, y_pred, y_pred_pro], axis=1)
    preAll = New_test[New_test.y_pred == 1]  # 可能会有重复的user_id
    preAll = preAll.sort_values(by='y_pred_pro', ascending=False)  # 根据概率值下降排序
    preAll = preAll.drop_duplicates(['user_id'])  # 默认保留第一个出现的(概率值大)的user_id

    predData_DataFrame = preAll[['user_id']]
    realData_DataFrame = realData_Zhong_Zhuan[['user_id']]

    predData_NpArray = np.array(predData_DataFrame, dtype=int)
    realData_NpArray = np.array(realData_DataFrame, dtype=int)

    predData = predData_NpArray.tolist()
    realData = realData_NpArray.tolist()

    lenPred = len(predData)
    lenReal = len(realData)
    print "线下测试集数据的个数：", lenReal
    print "线下预测的个数：", lenPred

    predUserIdData = [row[0] for row in predData]
    realUserIdData = [row[0] for row in realData]

    # print predUserIdData
    # print realUserIdData
    userIdCorrectNum = 0  # userId预测的正确数
    for userId in predUserIdData:
        if userId in realUserIdData:
            userIdCorrectNum += 1.0
    print 'userId正确个数：', userIdCorrectNum
    F11Precise = userIdCorrectNum / lenPred  # F11的正确率
    print 'F11Precise: ', F11Precise
    F11Recall = userIdCorrectNum / lenReal  # F11的召回率
    print 'F11Recall: ', F11Recall
    F11 = 6 * F11Recall * F11Precise / (5 * F11Recall + F11Precise)  # F11的值
    print 'F11:  ', F11

    return F11, predData_DataFrame, userIdCorrectNum, lenPred, F11Precise, F11Recall


def User_Score_Offline_New(y_pred, clf, X_test, test_Zhong_Zhuan, realData_Zhong_Zhuan, PreNum):
    "线下评分，按pro排序，指定前PreNum个数的user"
    y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
    y_pred_pro_tmp = clf.predict_proba(X_test)  # 算概率，去重用
    y_pred_pro = pd.DataFrame(y_pred_pro_tmp[:, 1], columns=['y_pred_pro'])  # 取y=1的概率

    New_test = pd.concat([test_Zhong_Zhuan, y_pred_pro], axis=1)
    preAll = New_test
    preAll = preAll.sort_values(by='y_pred_pro', ascending=False)  # 根据概率值下降排序
    preAll = preAll.drop_duplicates(['user_id'])  # 默认保留第一个出现的(概率值大)的user_id

    predData_DataFrame = preAll[['user_id']]
    predData_DataFrame = predData_DataFrame[0:PreNum]
    realData_DataFrame = realData_Zhong_Zhuan[['user_id']]

    predData_NpArray = np.array(predData_DataFrame, dtype=int)
    realData_NpArray = np.array(realData_DataFrame, dtype=int)

    predData = predData_NpArray.tolist()
    realData = realData_NpArray.tolist()

    lenPred = len(predData)
    lenReal = len(realData)
    print "线下测试集数据的个数：", lenReal
    print "线下预测的个数：", lenPred

    predUserIdData = [row[0] for row in predData]
    realUserIdData = [row[0] for row in realData]

    # print predUserIdData
    # print realUserIdData
    userIdCorrectNum = 0  # userId预测的正确数
    for userId in predUserIdData:
        if userId in realUserIdData:
            userIdCorrectNum += 1.0
    print 'userId正确个数：', userIdCorrectNum
    F11Precise = userIdCorrectNum / lenPred  # F11的正确率
    print 'F11Precise: ', F11Precise
    F11Recall = userIdCorrectNum / lenReal  # F11的召回率
    print 'F11Recall: ', F11Recall
    F11 = 6 * F11Recall * F11Precise / (5 * F11Recall + F11Precise)  # F11的值
    print 'F11的值: ', F11

    return F11, predData_DataFrame, userIdCorrectNum, lenPred, F11Precise, F11Recall