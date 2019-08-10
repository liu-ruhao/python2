




config={
#变量声明
"N_CLASSES":0,#数据类型
"IMG_W":64,#resize图像
"IMG_H":64,
"BATCH_SIZE" : 20,
"CAPACITY" : 200,
"MAX_STEP": 10000 , # 一般大于10K
"learning_rate" : 0.0001,  # 一般小于0.0001
"ratio":0.9,#训练集与测试集的比率
# 获取批次batch
"train_dir": 'D:/dali_tju/image_reg/day5/G_Flower/input_data/' , # 训练样本的读入路径
"logs_train_dir": 'D:/dali_tju/image_reg/day5/G_Flower/save' # logs存储路径
      }