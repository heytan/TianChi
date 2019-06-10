import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def prepare(dataset):
    """数据预处理
    1.折扣处理：
        判断折扣是“满减”还是“折扣率”，新增一列“is_manjian”表示该信息；
        将“满减”折扣转换为“折扣率”形式，新增一列“discount_rate”表示该信息；
        得到满减折扣的最低消费，新增一列“min_cost_of_manjian”表示该信息；
    2.距离处理：
        将空距离填充为-1；
        判断是否为空距离，新增一列“null_distance”表示该信息；
    3.时间处理方便计算时间差
        将“Date_received”列中int或float类型元素转换成datetime类型，新增一列“date_received”表示该信息；
        将“Date”列中int类型元素转换为datetime类型，新增一列“date”表示该信息；
    Args:
        dataset: off_train 和 off_test DataFrame类型；
    Returns:
        data 预处理后的DataFrame类型
    """
    #1.
    data = dataset.copy()
    data['is_manjian'] = data['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    data['discount_rate'] = data['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else (1-float(str(x).split(':')[1])/float(str(x).split(':')[0])))
    data['min_cost_of_manjian'] = data['Discount_rate'].map(lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))
    #2.
    data['Distance'].fillna(-1,inplace=True)
    data['null_distance'] = data['Distance'].map(lambda x: 1 if x == -1 else 0)
    #3.
    data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
    if 'Date' in data.columns.values:
        data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    
    return data

def get_label(dataset):
    """打标
    领券后15天之内使用为1，否则为0，新增一列“label”表示该信息
    Args:
        dataset: 
    Return:
        打标后的DataFrame
    """
    data = dataset.copy()
    data['label'] = list(map(lambda x, y: 1 if (x-y).total_seconds()/(60*60*24) <= 15 else 0, data['date'], data['date_received']))
    return data

def get_simple_feature(label_field):
    """提取的5个特征，作为初学示例
    1.'simple_User_id_receive_cnt':用户领券数量
    2.'simple_User_id_Coupon_id_receive_cnt':用户领取特定优惠券数量
    3.'simple_User_id_Date_received_receive_cnt':用户当天领券数
    4.'simple_User_id_Coupon_id_Date_received_receive_cnt':用户当天领取特定优惠券数
    5.'simple_User_id_Coupon_id_Date_received_repeat_receive':用户是否在同一天重复领取特定优惠券
    
    Args:
        label_field: DataFrame类型的数据集
    Returns:
        feature: 提取完特征后DataFrame类型的数据集
    """
    data = label_field.copy()
    #由于Coupon_id和Date_received含有缺失值，读入时是浮点型，需要转为整型。
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1
    
    feature = data.copy()
    
    #1.用户领券数
    keys = ['User_id'] #主键
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns = {'cnt':prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    
    #2.用户领取特定的优惠券数
    keys = ['User_id', 'Coupon_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns = {'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    
    #3.用户当天领券数
    keys = ['User_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns = {'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    
    #4.用户当天领取特定优惠券数
    keys = ['User_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns = {'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    
    #5.用户是否在同一天重复领取特定优惠券
    keys = ['User_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=lambda x: 1 if len(x) > 1 else 0)
    pivot = pd.DataFrame(pivot).rename(columns = {'cnt': prefixs + 'repeat_receive'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    
    feature.drop(['cnt'], axis=1, inplace=True)
    return feature      

def model_xgb(train, test):
    """xgboost模型
    调用xgboost模型进行训练预测
    Args:
        train: 训练集（包含label）
        test:测试集（不含label）
    Returns:
        result: 预测结果，包含User_id, Coupon_id, Date_received, prob, 其中prob表示预测为1的概率
        feat_importance: 特征重要性, 包含属性 feature_name, importance.
        model: 训练完的模型
    """
    params = {'booster': 'gbtree',
             'objective': 'binary:logistic',
             'eval_metric': 'auc',
             'silent': 1,
             'eta': 0.01,
             'max_depth': 5,
             'min_child_weight': 1,
             'gamma': 0,
             'lambda': 1,
             'colsample_bylevel': 0.7,
             'colsample_bytree':0.7,
             'subsample': 0.9,
             'scale_pos_weight': 1}
    
    #数据转模型可用格式
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis = 1), label= train['label'])
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    
    model = xgb.train(params, dtrain, num_boost_round = 5167)
    predict = model.predict(dtest)
    predict = pd.DataFrame(predict, columns = ['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis = 1)
    
    feat_importance = pd.DataFrame(columns = ['feature_name', 'importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'], ascending = False, inplace = True)
    
    return result, feat_importance

def off_evaluate(validate, off_result):
    """线下验证
    1.评测指标为AUC，但不是直接计算AUC，是对每个Coupon_id单独计算该核销预测AUC，再对所有优惠券的AUC值求平均作为最终的评价标准。
    2.注意计算AUC时同个Coupon_id必须有两类真值标签，所以先过滤全被核销的Coupon_id（该id标签真实值全为1）和全没被核销的Coupon_id（该id标签真实值全为0）
    Args:
        validate:验证集
        off_result:验证集的预测结果
    """
    evaluate_data = pd.concat([validate[['Coupon_id', 'label']], off_result[['prob']]], axis = 1)
    aucs = 0
    lens = 0
    for name, group in evaluate_data.groupby('Coupon_id'):
        if len(set(list(group['label']))) == 1:
            continue
        aucs += roc_auc_score(group['label'], group['prob'])
        lens += 1
    auc = aucs / lens
    return auc

if __name__ == '__main__':
	#读取数据
	off_train = pd.read_csv('datalab/20394/ccf_offline_stage1_train.csv')
	off_test = pd.read_csv('datalab/20394/ccf_offline_stage1_test_revised.csv')

	#数据预处理
	off_train = prepare(off_train)
	off_test = prepare(off_test)

	#打标签
	off_train = get_label(off_train)

	#划分数据集
	train = off_train[off_train['date_received'].isin(pd.date_range('2016/1/1', periods = 135))]
	validate = off_train[off_train['date_received'].isin(pd.date_range('2016/5/16', periods = 30))]


	#构造训练集、验证集、测试集
	print("构造训练集")
	train = get_simple_feature(train)[['User_id', 'Coupon_id', 'Date_received', 'is_manjian', 'discount_rate',
                                   'min_cost_of_manjian', 'null_distance', 'label', 'simple_User_id_receive_cnt', 
                                   'simple_User_id_Coupon_id_receive_cnt', 'simple_User_id_Date_received_receive_cnt', 
                                   'simple_User_id_Coupon_id_Date_received_receive_cnt', 
                                   'simple_User_id_Coupon_id_Date_received_repeat_receive']]

	print("构造验证集")
	validate = get_simple_feature(validate)[['User_id', 'Coupon_id', 'Date_received', 'is_manjian', 'discount_rate', 
                                         'min_cost_of_manjian', 'null_distance', 'label', 'simple_User_id_receive_cnt', 
                                         'simple_User_id_Coupon_id_receive_cnt', 'simple_User_id_Date_received_receive_cnt', 
                                         'simple_User_id_Coupon_id_Date_received_receive_cnt', 
                                         'simple_User_id_Coupon_id_Date_received_repeat_receive']]

	print("构造测试集")
	test = get_simple_feature(off_test)[['User_id', 'Coupon_id', 'Date_received', 'is_manjian', 'discount_rate', 
                                 'min_cost_of_manjian', 'null_distance', 'simple_User_id_receive_cnt', 
                                 'simple_User_id_Coupon_id_receive_cnt', 'simple_User_id_Date_received_receive_cnt', 
                                 'simple_User_id_Coupon_id_Date_received_receive_cnt', 
                                 'simple_User_id_Coupon_id_Date_received_repeat_receive']]



    #线下验证
    off_result, off_feat_importance = model_xgb(train, validate.drop(['label'], axis = 1))
	auc = off_evaluate(validate, off_result)
	print("线下验证的AUC = {}".format(auc))

	#线上验证
	big_train = pd.concat([train, validate], axis = 0)
	result, _ = model_xgb(big_train, test)
	result.to_csv('myspace/submit2.csv', index=False, header=False)

