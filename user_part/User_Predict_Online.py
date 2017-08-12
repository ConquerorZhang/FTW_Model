# -*- coding: utf-8 -*-

import pandas as pd

def User_Predict_Online(online,gbm):
    "预测线上数据，保存pro>0.5的user"
    online_columns = [x_ol for x_ol in online.columns if x_ol not in [ 'user_id']]
    X_online_tmp = online[online_columns]
    X_online = pd.get_dummies(X_online_tmp, prefix=['age', 'sex'],
                              columns=['age', 'sex'])
    
    Y_online = gbm.predict(X_online)  # 预测线上下单的
    y_online_pred_pro_tmp = gbm.predict_proba(X_online)  # 算概率，去重用
    y_online_pred_pro = pd.DataFrame(y_online_pred_pro_tmp[:, 1], columns=['y_pred_pro'])  # 取y=1的概率
    
    Y_online = pd.DataFrame(Y_online, columns=['Y_online'])
    New_online = pd.concat([online, Y_online, y_online_pred_pro], axis=1)
    preOnlineAll = New_online[New_online.Y_online == 1]  # 可能有重复的user_id
    preOnlineAll = preOnlineAll.sort_values(by='y_pred_pro', ascending=False)  # 根据概率值下降排序
    preOnlineAll = preOnlineAll.drop_duplicates(['user_id'])  # 默认保留第一个出现的(概率值大)的user_id
    
    preOnlineData = preOnlineAll[['user_id']]
    preOnlineData.to_csv('./user_CSV/Online_User.csv', index=False)  # 不带索引
    
    print "线上预测数据个数：", len(preOnlineData)
    return len(preOnlineData)


def User_Predict_Online_New(online, gbm, PreNum):
    "预测线上数据，保存所有的user和对应的pro"
    online_columns = [x_ol for x_ol in online.columns if x_ol not in ['user_id']]
    X_online_tmp = online[online_columns]
    X_online = pd.get_dummies(X_online_tmp, prefix=['age', 'sex'],
                              columns=['age', 'sex'])

    #Y_online = gbm.predict(X_online)  # 预测线上下单的
    y_online_pred_pro_tmp = gbm.predict_proba(X_online)  # 算概率，去重用
    y_online_pred_pro = pd.DataFrame(y_online_pred_pro_tmp[:, 1], columns=['y_pred_pro'])  # 取y=1的概率

    #Y_online = pd.DataFrame(Y_online, columns=['Y_online'])
    New_online = pd.concat([online, y_online_pred_pro], axis=1)
    preOnlineAll = New_online#[New_online.Y_online == 1]  # 可能有重复的user_id
    preOnlineAll = preOnlineAll.sort_values(by='y_pred_pro', ascending=False)  # 根据概率值下降排序
    preOnlineAll = preOnlineAll.drop_duplicates(['user_id'])  # 默认保留第一个出现的(概率值大)的user_id

    preOnlineData = preOnlineAll[['user_id','y_pred_pro']]
    preOnlineData = preOnlineData[0:PreNum]
    preOnlineData.to_csv('./user_CSV/Online_User.csv', index=False)  # 不带索引

    print "线上预测数据个数：", len(preOnlineData)
    return len(preOnlineData)