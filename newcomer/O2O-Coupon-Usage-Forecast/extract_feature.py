import pandas as pd 
import numpy as np 
from datetime import date 
#datetime这个包可以深入学习一下。

"""
dataset split:
			(date_received)
		dataset3: 20160701~20160731 (113640), features3 from 20160315~20160630 (off_test)
		dataset2: 20160515~20160615 (258446), features2 from 20160201~20160514
		dataset1: 20160414~20160514 (138303), features1 from 20160101~20160413

1. merchant related:
		sales_use_coupon. total_coupon
		transfer_rate = sales_use_coupon/total_coupon.
		merchant_avg_distance, merchant_min_distance, merchant_max_distance of those use coupon
		total_sales. coupon_rate = sales_use_coupon/total_sales.

2.

"""