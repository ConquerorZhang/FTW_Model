# -*- coding: utf-8 -*-

def UserGouZaoTrain(train_All,train_Neg_frac):
    "控制正负样本比例，构造训练数据"
    train_Positive = train_All[(train_All.y == 1)]  # 取正样本
    train_Negative = train_All[(train_All.y == 0)]  # 取负样本
    # train_Positive = train_Positive_1.sample(frac=1.2, replace=True)
    train_Negative_Part = train_Negative.sample(frac=train_Neg_frac, replace=False)  # 按frac比例随机采样,无放回
    print "训练集的正负样本比例: 1:%s" % (float(len(train_Negative_Part)) / len(train_Positive))
    train_All = train_Positive.append(train_Negative_Part, ignore_index=False)  # 合并，(False:原来的索引,默认false)
    train_All = train_All.sort_index()  # 按索引排序                    
    return train_All