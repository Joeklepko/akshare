import pandas as pd
import akshare as ak

file_name = './data/stock_all_tmp.csv'
data = pd.read_csv(file_name)
name_all_list = list(data['代码'])

count = 1
for name in name_all_list:
    print(name, count)
    count += 1
    try:
        data_get = ak.stock_zh_a_hist(name[2:], adjust='qfq')
        data_get.to_csv(f'./data/stock_all_daily/{name[2:]}.csv')
    except:
        pass
