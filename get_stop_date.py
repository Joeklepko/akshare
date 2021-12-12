import akshare as ak
import pandas as pd
import datetime

# def get_date_list(begin_date,end_date):
#     date_list = [x.strftime('%Y%m%d') for x in list(pd.date_range(start=begin_date, end=end_date))]
#     return date_list
#
# date_list = get_date_list('1999-01-01', '2021-12-15')

# for date in date_list:
# print(date)
def get_stop_all_info():
    date = '19990101'
    stock_tfp_em_df = ak.stock_tfp_em(date=date)

    stop_count = len(stock_tfp_em_df)
    stop_all = {}
    for i in range(stop_count):
        tmp = stock_tfp_em_df.iloc[i]
        code = tmp[1]
        start_time = str(tmp[3])
        if type(tmp[8]) == datetime.date:
            end_time = str(tmp[8])
        elif type(tmp[4]) == datetime.date:
            end_time = str(tmp[4])
        else:
            end_time = 'nan'
        if code not in stop_all:
            stop_all[code] = [(start_time, end_time)]
        else:
            print(code)
            stop_all[code].append((start_time, end_time))
    return stop_all
