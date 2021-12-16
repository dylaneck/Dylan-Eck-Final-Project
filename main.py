from numpy import str0
import utils

streaming_df = utils.open_json()
#print(streaming_df)
streaming_df = utils.ms_to_s(streaming_df)

streaming_df = utils.split_string(streaming_df)


streaming_df, new_day_col = utils.add_day_col(streaming_df)

utils.pool_time(streaming_df)


week_ser, day_ser, weekend_ser, weekday_ser, streaming_df, new_time_ser = utils.aggregate_data(streaming_df)
#streaming_df = utils.seperate_dates(streaming_df)
#streaming_df = utils.make_warm_cold(streaming_df)
streaming_df.to_csv("streaming.csv")
utils.make_chart(week_ser, day_ser)
utils.make_chart_2(weekend_ser)

days_ser = utils.pool_by_day(streaming_df, "day.csv")
days_ser = utils.seperate_dates(days_ser)
days_ser = utils.make_warm_cold(days_ser)
days_ser.to_csv("days_2.csv")
streaming_df.to_csv("streaming.csv")
week_df, wend_df, monday_df, tuesday_df, wednesday_df, thursday_df, friday_df, saturday_df, sunday_df = utils.make_sep_df(days_ser)
#utils.make_chart_3(days_ser)
#utils.run_t_test(week_df, wend_df)
#streaming_df = utils.seperate_time_data(streaming_df)

#streaming_df["endTime"] = new_time_ser

##one_ser, two_ser, three_ser, four_ser, five_ser, six_ser, section_ser, day_ser_2 = utils.split_time(streaming_df)
weeks_ser = utils.split_weeks(days_ser)
print(weeks_ser)
#days_ser.to_csv("days.csv")
#streaming_df.to_csv("streaming.csv")
##utils.make_chart_4(section_ser, day_ser_2)
##utils.days_ttest(monday_df, tuesday_df, wednesday_df, friday_df, saturday_df, sunday_df)


