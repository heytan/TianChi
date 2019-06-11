# 天池新人实战赛O2O优惠券使用预测  
## 赛题背景  
随着移动设备的完善和普及，移动互联网+各行各业进入了高速发展阶段，这其中以O2O（Online to Offline）消费最为吸引眼球。据不完全统计，O2O行业估值上亿的创业公司至少有10家，也不乏百亿巨头的身影。O2O行业天然关联数亿消费者，各类APP每天记录了超过百亿条用户行为和位置记录，因而成为大数据科研和商业化运营的最佳结合点之一。 以优惠券盘活老用户或吸引新客户进店消费是O2O的一种重要营销方式。然而随机投放的优惠券对多数用户造成无意义的干扰。对商家而言，滥发的优惠券可能降低品牌声誉，同时难以估算营销成本。 个性化投放是提高优惠券核销率的重要技术，它可以让具有一定偏好的消费者得到真正的实惠，同时赋予商家更强的营销能力。本次大赛为参赛选手提供了O2O场景相关的丰富数据，希望参赛选手通过分析建模，精准预测用户是否会在规定时间内使用相应优惠券。
## 数据和评价方式  
本赛题提供用户在2016年1月1日至2016年6月30日之间真实线上线下消费行为，预测用户在2016年7月领取优惠券后15天以内的使用情况。本赛题目标是预测投放的优惠券是否核销。针对此任务及一些相关背景知识，使用优惠券核销预测的平均AUC（ROC曲线下面积）作为评价标准。 即对每个优惠券coupon_id单独计算核销预测的AUC值，再对所有优惠券的AUC值求平均作为最终的评价标准。
## 1.数据初探  
获取对数据的基本了解 [data_explore.ipynb](https://github.com/heytan/TianChi/blob/master/newcomer/O2O-Coupon-Usage-Forecast/data_explore.ipynb)  
* 共有1754884条记录  
* 共有1053282条优惠券领取记录  
* 共有9738种不同优惠券  
* 共有539438个用户  
* 共有8417个商家  
* 优惠券在2016-01-01到2016-06-15之间领取  
* 消费时间在2016-01-01到2016-06-30之间  
## 2.数据分析与可视化
通过可视化方式深入了解数据各个变量之间的联系 [plot.ipynb](https://github.com/heytan/TianChi/blob/master/newcomer/O2O-Coupon-Usage-Forecast/plot.ipynb)  
### (1)每天领券次数  
![bar_1](https://raw.githubusercontent.com/heytan/TianChi/master/newcomer/O2O-Coupon-Usage-Forecast/imgs/bar_1.png)  
--- 
上图是一个从2016-01-01到2016-06-15期间每天被领券数量的柱状图，从图中可以看出1月份左右的数据明显高于其他时间，最高单日领券数达到了71658次。
### (2)每月各类消费折线图  
![line_1](https://raw.githubusercontent.com/heytan/TianChi/master/newcomer/O2O-Coupon-Usage-Forecast/imgs/line_1.png)  
---
上图是2016-01到2016-06之间的每个月的各类消费的折线图。从图上可以看出，1月领券数量和消费数量严重不匹配，后面几个月的数据都趋于正常更适合用于划分数据集。在划分数据集时可以多考虑使用后面几个月的数据作为训练集。
### (3)消费距离折线图  
![bar_2](https://raw.githubusercontent.com/heytan/TianChi/master/newcomer/O2O-Coupon-Usage-Forecast/imgs/bar_2.png)  
---
用户更倾向于在距离较近或距离很远的地方进行消费，距离很好理解，很多用户也喜欢去距离很远的店消费，可能是外出旅游之类的消费。
### (4)消费距离与核销率柱状图  
![bar_3](https://raw.githubusercontent.com/heytan/TianChi/master/newcomer/O2O-Coupon-Usage-Forecast/imgs/bar_3.png)  
---
核销率基本上随着距离的增加而降低。
### (5)各类消费券数量占比饼图  
![pie_1](https://raw.githubusercontent.com/heytan/TianChi/master/newcomer/O2O-Coupon-Usage-Forecast/imgs/pie_1.png)  
---
被领取的优惠券绝大部分是满减类型。
### (6)核销优惠券的占比图  
![pie_2](https://raw.githubusercontent.com/heytan/TianChi/master/newcomer/O2O-Coupon-Usage-Forecast/imgs/pie_2.png)  
---
满减优惠券依旧占据绝大部分比例。
### (7)各种折扣率的优惠券领取与核销柱状图  
![bar_4](https://raw.githubusercontent.com/heytan/TianChi/master/newcomer/O2O-Coupon-Usage-Forecast/imgs/bar_4.png)  
---
核销最多的折扣率是0.83左右，因此折扣率对核销会有明显的影响。
### (8)每周内领券数与核销数折线图  
![line_2](https://raw.githubusercontent.com/heytan/TianChi/master/newcomer/O2O-Coupon-Usage-Forecast/imgs/line_2.png)  
---
周末的领券数和核销数整体上是大于工作日的，周几也会对优惠券使用产生影响。
### (9)正负样本比例图  
![pie_3](https://raw.githubusercontent.com/heytan/TianChi/master/newcomer/O2O-Coupon-Usage-Forecast/imgs/pie_3.png)  
---
正负样本比例差距很大，后期建模时需啊哟考虑数据不平衡的问题。
## 3.数据集划分  
预测集：dataset3(20160701-20160731)(from off_test)+feature3(20160315-20160630)(from off_train)  
训练集：dataset2(20160515-206160615)(from off_train)+feature2(20160201-20160514)(from off_train)  
训练集：dataset1(20160414-20160514)(from off_train)+feature1(20160101-20160413)(from off_train) 

feature区间是用来提取历史特征的，比如用户的一些习惯特征/商户的营业习惯等，这些历史个性特征对于预测优惠券的使用有重要影响。这种数据划分方法叫做时间窗划分法。
## 4.特征工程  
* other feature  
    * this_month_user_receive_all_coupon_count：用户领取的所有优惠券数目
    * this_month_user_receive_same_coupon_count：用户领取特定优惠券数目
    * this_month_user_receive_same_coupon_lastone：用户是否最后一次领取特定优惠券
    * this_month_user_receive_same_coupon_firstone：用户是否首次领取优惠券
    * this_day_user_receive_all_coupon_count：用户当天领取的优惠券数目
    * this_day_user_receive_same_coupon_count：用户当天领取的特定优惠券数目
    * day_gap_before：用户上一次领取的时间间隔
    * day_gap_after：用户下一次领取的时间间隔
    * max_date_received:用户领取特定优惠券的最晚日期
    * min_date_received:用户领取特定优惠券的最早日期
* coupon related feature  
    * discount_rate：折扣率
    * discount_man：满多少  
    * discount_jian：减多少  
    * is_man_jian：是否是满减类型  
    * day_of_week：周几  
    * day_of_month：月份  
    * days_distance:距离6月底还有几天  
* merchant related feature  
    * total_sales：商家进行交易的次数
    * sales_use_coupon：商家用优惠券进行交易的次数
    * total_coupon：商家共有多少优惠券
    * coupon_rate = sales_use_coupon/total_sales：用券交易数/总交易数  
    * merchant_coupon_transfer_rate=sales_use_coupon/total_coupon:用券交易数/总交易数
    * merchant_median_distance：商家距离的中位数
    * merchant_mean_distance:商家平均距离
    * merchant_min_distance：商家最小距离
    * merchant_max_distance：商家最大距离
* user related feature    
    * count_merchant：用户去过的店的数量。
    * user_avg_distance：用户平均距离
    * user_min_distance：用户最小距离
    * user_max_distance：用户最大距离
    * buy_use_coupon：用户用券消费次数
    * buy_total：用户总消费次数
    * coupon_received：用户领券次数
    * user_date_datereceived_gap：领券到消费的时间间隔
    * avg_user_date_datereceived_gap:平均领券到消费时间间隔
    * min_user_date_datereceived_gap:最小领券到消费时间间隔
    * max_user_date_datereceived_gap:最大领券到消费事假间隔
* user-merchant feature  
    * user_merchant_buy_total:用户去特定店消费次数
    * user_merchant_received:用户领券特定店优惠券数量
    * user_merchant_buy_use_coupon：用户去特定店用券消费的次数
    * user_merchant_any：用户接触过特定店次数（包括领券、消费等）
    * user_merchant_buy_common：用户对特定商家普通消费的次数
    * user_merchant_coupon_transfer_rate：用户对特定商家核销次数占领券次数比例
    * user_merchant_coupon_buy_rate：用户对特定商家核销次数占消费次数比例
    * user_merchant_rate：用户对特定商家消费次数占接触次数的比例
    * user_merchant_common_buy_rate：用户对特定商家普通消费次数占总消费次数比例
[O2O_full_version_1.py](https://github.com/heytan/TianChi/blob/master/newcomer/O2O-Coupon-Usage-Forecast/O2O_full_version_1.py)是第一个代码版本，这个版本只使用了一些非常简单的特征，数据集使用简单留出法划分。 
[O2O_full_version_2.py](https://github.com/heytan/TianChi/blob/master/newcomer/O2O-Coupon-Usage-Forecast/O2O_full_version_2.py)是第二个代码版本，这个版本系统的使用了特征工程的技术，使用了各种各样的特征，且数据集的划分使用时间窗划分法。

## 算法选择与训练  
使用了XGBOOST模型来做训练，但是对XGBOOST算法的原理还不清楚，需要后续的学习。同时对训练过程中如何避免过拟合等其他技巧需要学习。打算学习《统计学习方法》。
## 总结：  
经过新人赛O2O项目对整个机器学习，尤其是特征工程，以及pandas的操作加深了理解。





