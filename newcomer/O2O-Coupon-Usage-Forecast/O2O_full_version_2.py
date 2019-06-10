import pandas as pd 
import numpy as np 
from datetime import date 
import xgboost as xgb 
from sklearn.preprocessing import MinMaxScaler


def is_firstlastone(x):
	if x==0:
		return 1
	elif x>0:
		return 0
	else:
		return -1

def get_day_gap_before(s):
	date_received, dates = s.split('-')
	dates = dates.split(':')
	gaps = []
	for d in dates:
		this_gap = (date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:])) - date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
		if this_gap>0:
			gaps.append(this_gap)
	if len(gaps)==0:
		return -1
	else:
		return min(gaps)

def get_day_gap_after(s):
	date_received, dates = s.split('-')
	dates = dates.split(':')
	gaps = []
	for d in dates:
		this_gap = (date(int(d[0:4]),int(d[4:6]),int(d[6:8]))-date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
		if this_gap>0:
			gaps.append(this_gap)
	if len(gaps)==0:
		return -1
	else:
		return min(gaps)

def calc_discount_rate(s):
	s = str(s)
	s = s.split(':')
	if len(s)==1:
		return float(s[0])
	else:
		return 1.0-float(s[1])/float(s[0])

def get_discount_man(s):
	s = str(s)
	s = s.split(':')
	if len(s) == 1:
		return 'null'
	else:
		return int(s[0])

def get_discount_jian(s):
	s = str(s)
	s = s.split(':')
	if len(s) == 1:
		return 'null'
	else:
		return int(s[1])

def is_man_jian(s):
	s = str(s)
	s = s.split(':')
	if len(s) == 1:
		return 0
	else:
		 return 1

def get_user_date_datereceived_gap(s):
    s = s.split(':')
    return (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8])) - date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days

def get_label(s):
	s = s.split(':')
	if s[0] == 'null':
		return 0
	elif (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days <= 15:
		return 1
	else:
		return -1


#对dataset进行操作,获取other feature
def other_feature(dataset, name):
	"""
	other feature:
      this_month_user_receive_all_coupon_count：用户领取的所有优惠券数目
      this_month_user_receive_same_coupon_count：用户领取特定优惠券数目
      this_month_user_receive_same_coupon_lastone：用户是否最后一次领取特定优惠券
      this_month_user_receive_same_coupon_firstone：用户是否首次领取优惠券
      this_day_user_receive_all_coupon_count：用户当天领取的优惠券数目
      this_day_user_receive_same_coupon_count：用户当天领取的特定优惠券数目
      day_gap_before：用户上一次领取的时间间隔
      day_gap_after：用户下一次领取的时间间隔
      max_date_received:用户领取特定优惠券的最晚日期
      min_date_received:用户领取特定优惠券的最早日期
	"""
	
	t = dataset[['user_id']]
	t['this_month_user_receive_all_coupon_count'] = 1
	t = t.groupby('user_id').agg('sum').reset_index()

	#这个月用户领取同种优惠券的数量
	t1 = dataset[['user_id', 'coupon_id']]
	t1['this_month_user_receive_same_coupon_count'] = 1
	t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

	#用户领取特定优惠券的最早最晚日期
	t2 = dataset[['user_id', 'coupon_id', 'date_received']]
	t2.date_received = t2.date_received.astype('str')
	t2 = t2.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
	t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
	t2 = t2[t2.receive_number>1]
	t2['max_date_received'] = t2.date_received.apply(lambda s: max([int(d) for d in s.split(':')]))
	t2['min_date_received'] = t2.date_received.apply(lambda s: min([int(d) for d in s.split(':')]))
	t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

	#这个月用户首次/最后一次领取特定优惠券
	t3 = dataset[['user_id', 'coupon_id', 'date_received']]
	t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left')
	t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received.astype('int')
	t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype('int') - t3.min_date_received


	t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
	t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(is_firstlastone)
	t3 = t3[['user_id', 'coupon_id', 'date_received', 'this_month_user_receive_same_coupon_lastone', 'this_month_user_receive_same_coupon_firstone']]

	#一天中用户领券数量
	t4 = dataset[['user_id', 'date_received']]
	t4['this_day_user_receive_all_coupon_count'] = 1
	t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

	#一天中用户领取同类优惠券数量
	t5 = dataset[['user_id', 'coupon_id', 'date_received']]
	t5['this_day_user_receive_same_coupon_count'] = 1
	t5 = t5.groupby(['user_id', 'coupon_id', 'date_received']).agg('sum').reset_index()

	#用户领特定券的时间点
	t6 = dataset[['user_id', 'coupon_id', 'date_received']]
	t6.date_received = t6.date_received.astype('str')
	t6 = t6.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
	t6.rename(columns={'date_received':'dates'}, inplace=True)

	#用户领取特定优惠券之前几天曾领取过，之后几天将再次领取。
	t7 = dataset[['user_id', 'coupon_id', 'date_received']]
	t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left')
	t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
	t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
	t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
	t7 = t7[['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after']]

	other_feature = pd.merge(t1, t, on=['user_id'])
	other_feature = pd.merge(other_feature, t3, on=['user_id', 'coupon_id'])
	other_feature = pd.merge(other_feature, t4, on=['user_id', 'date_received'])
	other_feature = pd.merge(other_feature, t5, on=['user_id', 'coupon_id', 'date_received'])
	other_feature = pd.merge(other_feature, t7, on=['user_id', 'coupon_id', 'date_received'])
	other_feature.to_csv('data/'+ name + '.csv', index=None)
	print other_feature.shape

#对dataset进行操作，获取coupon related feature
#	discount_rate：折扣率
#	discount_man ：满多少
#	discount_jian：减多少
#	is_man_jian：是否是满减类型
#   day_of_week：周几
#   day_of_month：月份
#   days_distance:距离6月底还有几天 
#   (date_received)
def coupon_related(dataset, name):
	dataset['day_of_week'] = dataset.date_received.astype('str').apply(lambda x: date(int(x[0:4]),int(x[4:6]), int(x[6:8])).weekday()+1)
	dataset['day_of_month'] = dataset.date_received.astype('str').apply(lambda x: int(x[6:8]))
	dataset['days_distance'] = dataset.date_received.astype('str').apply(lambda x:(date(int(x[0:4]),int(x[4:6]),int(x[6:8]))-date(2016,6,30)).days)
	dataset['discount_man'] = dataset.discount_rate.apply(get_discount_man)
	dataset['discount_jian'] = dataset.discount_rate.apply(get_discount_jian)
	dataset['is_man_jian'] = dataset.discount_rate.apply(is_man_jian)
	dataset['discount_rate'] = dataset.discount_rate.apply(calc_discount_rate)
	d = dataset[['coupon_id']]
	d['coupon_count'] = 1
	d = d.groupby('coupon_id').agg('sum').reset_index()
	dataset = pd.merge(dataset, d, on='coupon_id', how='left')
	dataset.to_csv('data/' + name + '.csv', index=None)

"""
对featue区间进行操作，获取merchant related feature: 
      total_sales：商家进行交易的次数
      sales_use_coupon：商家用优惠券进行交易的次数
      total_coupon：商家共有多少优惠券
      coupon_rate = sales_use_coupon/total_sales：用券交易数/总交易数  
      merchant_coupon_transfer_rate = sales_use_coupon/total_coupon：用券交易数/总交易数
      merchant_median_distance：商家距离的中位数
      merchant_mean_distance:商家平均距离
      merchant_min_distance：商家最小距离
      merchant_max_distance：商家最大距离
"""
def merchant_related(feature, name):
	merchant = feature[['merchant_id', 'coupon_id', 'distance', 'date_received', 'date']]

	t = merchant[['merchant_id']]
	t.drop_duplicates(inplace=True)

	t1 = merchant[merchant.date != 'null'][['merchant_id']]
	t1['total_sales'] = 1
	t1 = t1.groupby('merchant_id').agg('sum').reset_index()

	t2 = merchant[(merchant.date != 'null')&(merchant.coupon_id != 'null')][['merchant_id']]
	t2['sales_use_coupon'] = 1
	t2 = t2.groupby('merchant_id').agg('sum').reset_index()

	t3 = merchant[merchant.coupon_id != 'null'][['merchant_id']]
	t3['total_coupon'] = 1
	t3 = t3.groupby('merchant_id').agg('sum').reset_index()

	t4 = merchant[(merchant.date != 'null')&(merchant.coupon_id != 'null')][['merchant_id', 'distance']]
	t4.replace('null', -1, inplace=True)
	t4.distance = t4.distance.astype('int')
	t4.replace(-1,np.nan,inplace=True)
	t5 = t4.groupby('merchant_id').agg('min').reset_index()
	t5.rename(columns={'distance':'merchant_min_distance'},inplace=True)

	t6 = t4.groupby('merchant_id').agg('max').reset_index()
	t6.rename(columns={'distance':'merchant_max_distance'}, inplace=True)

	t7 = t4.groupby('merchant_id').agg('mean').reset_index()
	t7.rename(columns={'distance':'merchant_mean_distance'}, inplace=True)

	t8 = t4.groupby('merchant_id').agg('median').reset_index()
	t8.rename(columns={'distance':'merchant_median_distance'}, inplace=True)

	merchant_feature = pd.merge(t,t1,on='merchant_id',how='left')
	merchant_feature = pd.merge(merchant_feature,t2,on='merchant_id',how='left')
	merchant_feature = pd.merge(merchant_feature,t3,on='merchant_id',how='left')
	merchant_feature = pd.merge(merchant_feature,t5,on='merchant_id',how='left')
	merchant_feature = pd.merge(merchant_feature,t6,on='merchant_id',how='left')
	merchant_feature = pd.merge(merchant_feature,t7,on='merchant_id',how='left')
	merchant_feature = pd.merge(merchant_feature,t8,on='merchant_id',how='left')
	merchant_feature.sales_use_coupon = merchant_feature.sales_use_coupon.replace(np.nan,0) #fillna with 0
	merchant_feature['merchant_coupon_transfer_rate'] = merchant_feature.sales_use_coupon.astype('float') / merchant_feature.total_coupon
	merchant_feature['coupon_rate'] = merchant_feature.sales_use_coupon.astype('float') / merchant_feature.total_sales
	merchant_feature.total_coupon = merchant_feature.total_coupon.replace(np.nan,0) #fillna with 0
	merchant_feature.to_csv('data/' + name + '.csv',index=None)


"""
user related: 
      count_merchant：用户去过的店的数量。
      user_avg_distance：用户平均距离
      user_min_distance：用户最小距离
      user_max_distance：用户最大距离
      buy_use_coupon：用户用券消费次数
      buy_total：用户总消费次数
      coupon_received：用户领券次数
      user_date_datereceived_gap：领券到消费的时间间隔
      avg_user_date_datereceived_gap:平均领券到消费时间间隔
      min_user_date_datereceived_gap:最小领券到消费时间间隔
      max_user_date_datereceived_gap:最大领券到消费事假间隔
      
"""
def user_related(feature, name):
	user = feature[['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']]

	t = user[['user_id']]
	t.drop_duplicates(inplace=True)

	t1 = user[user.date!='null'][['user_id','merchant_id']]
	t1.drop_duplicates(inplace=True)
	t1.merchant_id = 1
	t1 = t1.groupby('user_id').agg('sum').reset_index()
	t1.rename(columns={'merchant_id':'count_merchant'},inplace=True)

	t2 = user[(user.date!='null')&(user.coupon_id!='null')][['user_id','distance']]
	t2.replace('null',-1,inplace=True)
	t2.distance = t2.distance.astype('int')
	t2.replace(-1,np.nan,inplace=True)
	t3 = t2.groupby('user_id').agg('min').reset_index()
	t3.rename(columns={'distance':'user_min_distance'},inplace=True)

	t4 = t2.groupby('user_id').agg('max').reset_index()
	t4.rename(columns={'distance':'user_max_distance'},inplace=True)

	t5 = t2.groupby('user_id').agg('mean').reset_index()
	t5.rename(columns={'distance':'user_mean_distance'},inplace=True)

	t6 = t2.groupby('user_id').agg('median').reset_index()
	t6.rename(columns={'distance':'user_median_distance'},inplace=True)

	t7 = user[(user.date!='null')&(user.coupon_id!='null')][['user_id']]
	t7['buy_use_coupon'] = 1
	t7 = t7.groupby('user_id').agg('sum').reset_index()

	t8 = user[user.date!='null'][['user_id']]
	t8['buy_total'] = 1
	t8 = t8.groupby('user_id').agg('sum').reset_index()

	t9 = user[user.coupon_id!='null'][['user_id']]
	t9['coupon_received'] = 1
	t9 = t9.groupby('user_id').agg('sum').reset_index()

	t10 = user[(user.date_received!='null')&(user.date!='null')][['user_id','date_received','date']]
	t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
	t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
	t10 = t10[['user_id','user_date_datereceived_gap']]

	t11 = t10.groupby('user_id').agg('mean').reset_index()
	t11.rename(columns={'user_date_datereceived_gap':'avg_user_date_datereceived_gap'},inplace=True)
	t12 = t10.groupby('user_id').agg('min').reset_index()
	t12.rename(columns={'user_date_datereceived_gap':'min_user_date_datereceived_gap'},inplace=True)
	t13 = t10.groupby('user_id').agg('max').reset_index()
	t13.rename(columns={'user_date_datereceived_gap':'max_user_date_datereceived_gap'},inplace=True)


	user_feature = pd.merge(t,t1,on='user_id',how='left')
	user_feature = pd.merge(user_feature,t3,on='user_id',how='left')
	user_feature = pd.merge(user_feature,t4,on='user_id',how='left')
	user_feature = pd.merge(user_feature,t5,on='user_id',how='left')
	user_feature = pd.merge(user_feature,t6,on='user_id',how='left')
	user_feature = pd.merge(user_feature,t7,on='user_id',how='left')
	user_feature = pd.merge(user_feature,t8,on='user_id',how='left')
	user_feature = pd.merge(user_feature,t9,on='user_id',how='left')
	user_feature = pd.merge(user_feature,t11,on='user_id',how='left')
	user_feature = pd.merge(user_feature,t12,on='user_id',how='left')
	user_feature = pd.merge(user_feature,t13,on='user_id',how='left')
	user_feature.count_merchant = user_feature.count_merchant.replace(np.nan,0)
	user_feature.buy_use_coupon = user_feature.buy_use_coupon.replace(np.nan,0)
	user_feature['buy_use_coupon_rate'] = user_feature.buy_use_coupon.astype('float') / user_feature.buy_total.astype('float')
	user_feature['user_coupon_transfer_rate'] = user_feature.buy_use_coupon.astype('float') / user_feature.coupon_received.astype('float')
	user_feature.buy_total = user_feature.buy_total.replace(np.nan,0)
	user_feature.coupon_received = user_feature.coupon_received.replace(np.nan,0)
	user_feature.to_csv('data/' + name +'.csv',index=None)

"""
user_merchant:
      user_merchant_buy_total:用户去特定店消费次数
      user_merchant_received:用户领券特定店优惠券数量
      user_merchant_buy_use_coupon：用户去特定店用券消费的次数
      user_merchant_any：用户接触过特定店次数（包括领券、消费等）
      user_merchant_buy_common：用户对特定商家普通消费的次数
      user_merchant_coupon_transfer_rate：用户对特定商家核销次数占领券次数比例
      user_merchant_coupon_buy_rate：用户对特定商家核销次数占消费次数比例
      user_merchant_rate：用户对特定商家消费次数占接触次数的比例
      user_merchant_common_buy_rate：用户对特定商家普通消费次数占总消费次数比例
"""
def user_merchant(feature, name):
	all_user_merchant = feature[['user_id','merchant_id']]
	all_user_merchant.drop_duplicates(inplace=True)

	t = feature[['user_id','merchant_id','date']]
	t = t[t.date!='null'][['user_id','merchant_id']]
	t['user_merchant_buy_total'] = 1
	t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
	t.drop_duplicates(inplace=True)

	t1 = feature[['user_id','merchant_id','coupon_id']]
	t1 = t1[t1.coupon_id!='null'][['user_id','merchant_id']]
	t1['user_merchant_received'] = 1
	t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
	t1.drop_duplicates(inplace=True)

	t2 = feature[['user_id','merchant_id','date','date_received']]
	t2 = t2[(t2.date!='null')&(t2.date_received!='null')][['user_id','merchant_id']]
	t2['user_merchant_buy_use_coupon'] = 1
	t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
	t2.drop_duplicates(inplace=True)

	t3 = feature[['user_id','merchant_id']]
	t3['user_merchant_any'] = 1
	t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
	t3.drop_duplicates(inplace=True)

	t4 = feature[['user_id','merchant_id','date','coupon_id']]
	t4 = t4[(t4.date!='null')&(t4.coupon_id=='null')][['user_id','merchant_id']]
	t4['user_merchant_buy_common'] = 1
	t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
	t4.drop_duplicates(inplace=True)

	user_merchant = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
	user_merchant = pd.merge(user_merchant,t1,on=['user_id','merchant_id'],how='left')
	user_merchant = pd.merge(user_merchant,t2,on=['user_id','merchant_id'],how='left')
	user_merchant = pd.merge(user_merchant,t3,on=['user_id','merchant_id'],how='left')
	user_merchant = pd.merge(user_merchant,t4,on=['user_id','merchant_id'],how='left')
	user_merchant.user_merchant_buy_use_coupon = user_merchant.user_merchant_buy_use_coupon.replace(np.nan,0)
	user_merchant.user_merchant_buy_common = user_merchant.user_merchant_buy_common.replace(np.nan,0)
	user_merchant['user_merchant_coupon_transfer_rate'] = user_merchant.user_merchant_buy_use_coupon.astype('float') / user_merchant.user_merchant_received.astype('float')
	user_merchant['user_merchant_coupon_buy_rate'] = user_merchant.user_merchant_buy_use_coupon.astype('float') / user_merchant.user_merchant_buy_total.astype('float')
	user_merchant['user_merchant_rate'] = user_merchant.user_merchant_buy_total.astype('float') / user_merchant.user_merchant_any.astype('float')
	user_merchant['user_merchant_common_buy_rate'] = user_merchant.user_merchant_buy_common.astype('float') / user_merchant.user_merchant_buy_total.astype('float')
	user_merchant.to_csv('data/' + name + '.csv',index=None)

# generate training and testing set
def generate_train_test_set(file_coupon, file_merchant, file_user, file_user_merchant, file_other,file_save, add_label):
	coupon = pd.read_csv('data/' + file_coupon + '.csv', header = 0, keep_default_na=False)
	merchant = pd.read_csv('data/' + file_merchant + '.csv', header = 0, keep_default_na=False)
	user = pd.read_csv('data/' + file_user + '.csv', header = 0, keep_default_na=False)
	user_merchant = pd.read_csv('data/' + file_user_merchant + '.csv', header = 0, keep_default_na=False)
	other_feature = pd.read_csv('data/' + file_other + '.csv', header = 0, keep_default_na=False)
	dataset = pd.merge(coupon,merchant,on='merchant_id',how='left')
	dataset = pd.merge(dataset,user,on='user_id',how='left')
	dataset = pd.merge(dataset,user_merchant,on=['user_id','merchant_id'],how='left')
	dataset = pd.merge(dataset,other_feature,on=['user_id','coupon_id','date_received'],how='left')
	dataset.drop_duplicates(inplace=True)
	print dataset.shape

	dataset.user_merchant_buy_total = dataset.user_merchant_buy_total.replace(np.nan,0)
	dataset.user_merchant_any = dataset.user_merchant_any.replace(np.nan,0)
	dataset.user_merchant_received = dataset.user_merchant_received.replace(np.nan,0)
	dataset['is_weekend'] = dataset.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
	weekday_dummies = pd.get_dummies(dataset.day_of_week)
	weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
	dataset = pd.concat([dataset,weekday_dummies],axis=1)

	if not add_label:
		dataset.drop(['merchant_id','day_of_week','coupon_count'],axis=1,inplace=True)
	else:
		dataset['label'] = dataset.date.astype('str') + ':' + dataset.date_received.astype('str')
		dataset.label = dataset.label.apply(get_label)
		dataset.drop(['merchant_id', 'day_of_week', 'date', 'date_received', 'coupon_id', 'coupon_count'], axis=1, inplace=True)
	
	dataset = dataset.replace('null',np.nan)
	dataset.to_csv('data/' + file_save + '.csv',index=None)


if __name__ == '__main__':
	"""数据集划分
1.将2016年1月1日到4月13日的数据提取特征，4月14号到5月14号的做训练集2
2.将2016年2月1号到5月14号的数据提取特征，5月15号到6月15号的做训练集1
3.将2016年3月15号到6月30号的数据提取特征，7月1号到7月31号的做测试集

1.商户相关特征：

"""
#读取数据，更换列名字
off_train = pd.read_csv('data/ccf_offline_stage1_train.csv', header = 0, keep_default_na=False)
off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']

off_test = pd.read_csv('data/ccf_offline_stage1_test_revised.csv', header = 0, keep_default_na=False)
off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']

on_train = pd.read_csv('data/ccf_online_stage1_train.csv', header = 0, keep_default_na=False)
on_train.columns = ['user_id', 'merchant_id', 'action', 'coupon_id', 'discount_rate', 'date_received', 'date']

#过滤条件对空值有什么影响呢？
dataset3 = off_test
feature3 = off_train[((off_train.date>='20160315')&(off_train.date<='20160630'))|((off_train.date=='null')&(off_train.date_received>='20160315')&(off_train.date_received<='20160630'))]
dataset2 = off_train[(off_train.date_received>='20160515')&(off_train.date_received<='20160615')]
feature2 = off_train[(off_train.date>='20160201')&(off_train.date<='20160514')|((off_train.date=='null')&(off_train.date_received>='20160201')&(off_train.date_received<='20160514'))]
dataset1 = off_train[(off_train.date_received>='20160414')&(off_train.date_received<='20160514')]
feature1 = off_train[(off_train.date>='20160101')&(off_train.date<='20160413')|((off_train.date=='null')&(off_train.date_received>='20160101')&(off_train.date_received<='20160413'))]

"""
	other feature:
      this_month_user_receive_all_coupon_count
      this_month_user_receive_same_coupon_count
      this_month_user_receive_same_coupon_lastone
      this_month_user_receive_same_coupon_firstone
      this_day_user_receive_all_coupon_count
      this_day_user_receive_same_coupon_count
      day_gap_before, day_gap_after  (receive the same coupon)
"""
other_feature(dataset3, 'other_feature3')
other_feature(dataset2, 'other_feature2')
other_feature(dataset1, 'other_feature1')


"""
coupon related: 
      discount_rate. discount_man. discount_jian. is_man_jian
      day_of_week,day_of_month. (date_received)
"""
coupon_related(dataset3, 'coupon3_feature')
coupon_related(dataset2, 'coupon2_feature')
coupon_related(dataset1, 'coupon1_feature')

"""
merchant related: 
      total_sales. sales_use_coupon.  total_coupon
      coupon_rate = sales_use_coupon/total_sales.  
      transfer_rate = sales_use_coupon/total_coupon. 
      merchant_avg_distance,merchant_min_distance,merchant_max_distance of those use coupon
"""
merchant_related(feature3, 'merchant3_feature')
merchant_related(feature2, 'merchant2_feature')
merchant_related(feature1, 'merchant1_feature')

"""
user related: 
      count_merchant. 
      user_avg_distance, user_min_distance,user_max_distance. 
      buy_use_coupon. buy_total. coupon_received.
      buy_use_coupon/coupon_received. 
      buy_use_coupon/buy_total
      user_date_datereceived_gap
      
"""
user_related(feature3, 'user3_feature')
user_related(feature2, 'user2_feature')
user_related(feature1, 'user1_feature')

"""
user_merchant:
      times_user_buy_merchant_before. 
"""
user_merchant(feature3, 'user_merchant3')
user_merchant(feature2, 'user_merchant2')
user_merchant(feature1, 'user_merchant1')


## generate training and testing set
generate_train_test_set('coupon3_feature', 'merchant3_feature', 'user3_feature', 'user_merchant3', 'other_feature3', 'dataset3', True)
generate_train_test_set('coupon2_feature', 'merchant2_feature', 'user2_feature', 'user_merchant2', 'other_feature2', 'dataset2', False)
generate_train_test_set('coupon1_feature', 'merchant1_feature', 'user1_feature', 'user_merchant1', 'other_feature1', 'dataset1', False)


## 读取数据集准备训练模型
dataset1 = pd.read_csv('data/dataset1.csv')
dataset1.label.replace(-1, 0, inplace=True)
dataset2 = pd.read_csv('data/dataset2.csv')
dataset2.label.replace(-1, 0, inplace=True)
dataset3 = pd.read_csv('data/dataset3.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)

dataset12 = pd.concat([dataset1, dataset2], axis=0)

dataset1_y = dataset1.label
dataset1_x = dataset1.drop(['user_id', 'label', 'day_gap_before', 'day_gap_after'], axis = 1)
dataset2_y = dataset2.label
dataset2_x = dataset2.drop(['user_id', 'label', 'day_gap_before', 'day_gap_after'], axis= 1)
dataset12_y = dataset12.label
dataset12_x = dataset12.drop(['user_id', 'label', 'day_gap_before', 'day_gap_after'], axis= 1)
dataset3_preds = dataset3[['user_id','coupon_id','date_received']]
dataset3_x = dataset3.drop(['user_id','coupon_id','date_received','day_gap_before','day_gap_after'],axis=1)

print dataset1_x.shape
print dataset2_x.shape
print dataset3_x.shape


#转换为模型可识别数据格式
dataset1 = xgb.DMatrix(dataset1_x, label=dataset1_y)
dataset2 = xgb.DMatrix(dataset2_x, label=dataset2_y)
dataset12 = xgb.DMatrix(dataset12_x, label=dataset12_y)
dataset3 = xgb.DMatrix(dataset3_x)

params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }

watchlist = [(dataset12, 'train')]
model = xgb.train(params, dataset12, num_boost_round=3500, evals=watchlist)

dataset3_preds['label'] = model.predict(dataset3)
dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label.reshape(-1,1))
dataset3_preds.sort_values(by=['coupon_id', 'label'], inplace=True)
dataset3_preds.to_csv('data/submit2.csv', index=None, header=None)











