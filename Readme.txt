模型部分：
             开发语言       算法模型
            Python 2.7    lightgbm 0.1
工具：pycharm，Visual Stdio(lightgbm需要C++编译环境支持)
python类库：在Python中需要pandas、numpy基础包，LightGBM-master是LightGBM安装包，
官方安装教程参考：https://github.com/Microsoft/LightGBM/wiki/Installation-Guide。

1、运行：
在pycharm中运行Main()函数，最后会在user_sku_CSV文件夹下生成predict_Online.csv提交文件。
由于数据量大，并且在寻找较好的训练集时，有寻优采样的过程，所以时间上快则几十分钟，
慢则几个小时

2、程序主要框架：
Main()
  |-User_model() 训练User模型，预测online有可能下单的User
      |-UserGouZaoTrain() 通过负样本采样，构造训练集
      |-User_Score_Offline() 线下测试集评分
      |-User_Predict_Online() 预测线上可能下单的User
  |-User_Sku_Offline() 通过线下测试集训练User-Sku模型
      |-gouZaoTrain() 通过负样本采样，构造User-Sku训练集
      |-Score_Offline() 线下User-Sku测试集评分
  |-User_Sku_Online() 预测online会下单User的Sku

3、程序思路：
程序主要分两部分，一是User模型，另一个是User-Sku模型
User模型：训练集是正样本结合采样的负样本构造的，先默认模型参数，寻优训练集负样本的采样系数，
固定一个较好的采样系数；然后寻优模型参数，通过线下测试集的评分，固定一组较好的模型参数；
在所有参数固定后，用while循环负样本采样，结合线下测试集的评分指标，固定一组较好的训练样本，
并保存online预测的User。

User_Sku模型：线下和User模型类似，训练集是正样本结合采样的负样本构造的，先默认模型参数，寻优训
练集负样本的采样系数，固定一个较好的采样系数；然后寻优模型参数，通过线下测试集的评分，固定
一组较好的模型参数；在所有参数固定后，用while循环负样本采样，结合线下测试集的评分指标，固定
一组较好的训练样本；用固定好的训练样本训练User-Sku模型，用Online的全部User数据结合User模型
保存的User筛选出这些User的数据，作用到模型，输出User和对应的下单Sku

   

