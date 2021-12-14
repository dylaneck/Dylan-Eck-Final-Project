from numpy import str0
import utils

streaming_df = utils.open_json()

streaming_df = utils.ms_to_s(streaming_df)

streaming_df = utils.split_string(streaming_df)


streaming_df, new_day_col = utils.add_day_col(streaming_df)

utils.pool_time(streaming_df)


week_ser, day_ser, weekend_ser, weekday_ser, streaming_df = utils.aggregate_data(streaming_df)
streaming_df.to_csv("streaming.csv")
utils.make_chart(week_ser, day_ser)
utils.make_chart_2(weekend_ser)
week_df, wend_df = utils.make_sep_df(streaming_df)
days_ser,week_2, wend_2 = utils.pool_by_day(streaming_df)
print(week_2)
utils.make_chart_3(days_ser)
days_ser.to_csv("days.csv")
#utils.run_t_test(week_df, wend_df)