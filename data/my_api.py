import pandas as pd
import akshare as ak
import datetime

#all stock daily, constrain
data_all = ak.stock_zh_a_spot()
data_all.to_csv('stock_all_tmp.csv')
data_all.to_excel('stock_all_tmp.xlsx')
data_all.to_json('stock_all_tmp.json')
#无限制
ak.stock_zh_a_hist()
print(stock_zh_ah_name_dict)