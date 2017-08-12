# -*- coding: utf-8 -*-

from user_part import User_model
from user_sku_part import User_Sku_Offline
from user_sku_part import User_Sku_Online

#训练User模型，预测online有可能下单的User
User_model.User_model()

#通过线下测试集训练User-Sku模型
User_Sku_Offline.User_Sku_Offline()

#预测online会下单User的Sku
User_Sku_Online.User_Sku_Online()