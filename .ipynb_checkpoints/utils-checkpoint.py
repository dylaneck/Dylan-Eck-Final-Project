import json
from numpy.lib.shape_base import expand_dims
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def open_json():
    df_0 = pd.read_json("StreamingHistory0.json")
    df_1 = pd.read_json("StreamingHistory1.json")
    df_2 = pd.read_json("StreamingHistory2.json")
    df_3 = pd.read_json("StreamingHistory3.json")
    df_4 = pd.read_json("StreamingHistory4.json")
    df_5 = pd.read_json("StreamingHistory5.json")

    streaming_df = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5], ignore_index=True)

    return streaming_df

def ms_to_s(streaming_df):
    ms_ser = streaming_df["msPlayed"]
    s_ser = ms_ser.divide(1000)
    streaming_df["sPlayed"] = s_ser

    return streaming_df

def split_string(streaming_df):
    end_str = streaming_df["endTime"]
    new_ser = end_str.str.split(expand=True)
    streaming_df["date"] = new_ser[0]
    streaming_df["endTime"] = new_ser[1]
    
    return streaming_df


def add_day_col(streaming_df):
    date_ser = pd.Series(streaming_df["date"])
    new_date_ser = pd.Series()
    new_date_ser_2 = pd.Series()
    
    new_date_ser = pd.to_datetime(date_ser)
    new_date_ser_2 = new_date_ser.dt.day_name()
    streaming_df["Day_of_Week"] = new_date_ser_2
    return streaming_df, new_date_ser_2

def pool_time(streaming_df):
    s_ser = streaming_df["sPlayed"]
    total_sec = 0
    for i in range(len(s_ser)):
        total_sec = total_sec + s_ser[i]
    minutes = total_sec / 60
    hours = minutes / 60
    days = hours / 24
    print("Total Seconds:", total_sec, "Total minutes:", minutes, "Total hours:", hours, "Total days:", days)

def aggregate_data(streaming_df):
    week_ser = pd.Series()
    time_ser = pd.Series(streaming_df["sPlayed"])
    day_ser = pd.Series(streaming_df["Day_of_Week"])
    weekend_ser = pd.Series()
    weekday_ser = pd.Series(streaming_df["msPlayed"])
    monday = 0
    tuesday = 0
    wednesday = 0
    thursday = 0
    friday = 0
    saturday = 0
    sunday = 0
    weekdays = 0
    weekends  = 0

    for i in range(len(streaming_df)):
        if day_ser[i] == "Monday":
            monday += time_ser[i]
        elif day_ser[i] == "Tuesday":
            tuesday += time_ser[i]
        elif day_ser[i] == "Wednesday":
            wednesday += time_ser[i]
        elif day_ser[i] == "Thursday":
            thursday += time_ser[i]
        elif day_ser[i] == "Friday":
            friday += time_ser[i]
        elif day_ser[i] == "Saturday":
            saturday += time_ser[i]
        elif day_ser[i] == "Sunday":
            sunday += time_ser[i]

    for i in range(len(streaming_df)):
        if day_ser[i] == "Monday" or day_ser[i] == "Tuesday" or day_ser[i] == "Wednesday" or day_ser[i] == "Thursday" or day_ser[i] == "Friday":
            weekdays += time_ser[i]
        elif  day_ser[i] == "Saturday" or day_ser[i] == "Sunday":
            weekends += time_ser[i]
    for i in range(len(streaming_df)):
        if day_ser[i] == "Monday" or day_ser[i] == "Tuesday" or day_ser[i] == "Wednesday" or day_ser[i] == "Thursday" or day_ser[i] == "Friday":
            weekday_ser[i] = "Weekday"
        elif  day_ser[i] == "Saturday" or day_ser[i] == "Sunday":
            weekday_ser[i] = "Weekend"
    
    streaming_df["Weekdays"] = weekday_ser
    adj_weekdays = weekdays / 5
    adj_weekends = weekends / 2
    weekend_ser["Weekdays"] = adj_weekdays
    weekend_ser["Weekends"] = adj_weekends
    week_ser["Monday"] = monday
    week_ser["Tuesday"] = tuesday
    week_ser["Wednesday"] = wednesday
    week_ser["Thursday"] = thursday
    week_ser["Friday"] = friday
    week_ser["Saturday"] = saturday
    week_ser["Sunday"] = sunday

    Day_ser = pd.Series()
    Day_ser["Monday"] = "Monday"
    Day_ser["Tuesday"] = "Tuesday"
    Day_ser["Wednesday"] = "Wednesday"
    Day_ser["Thursday"] = "Thursday"
    Day_ser["Friday"] = "Friday"
    Day_ser["Saturday"] = "Saturday"
    Day_ser["Sunday"] = "Sunday"
    time_ser = streaming_df["endTime"]
    print(time_ser)

    return(week_ser, Day_ser, weekend_ser, weekday_ser, streaming_df, time_ser)

def make_chart(week_ser, day_ser):
    plt.figure()
    plt.bar(day_ser, week_ser, color="green")
    plt.xlabel("Day of Week")
    plt.ylabel("Number of Seconds of Music Listened to")
    plt.title("Number of Seconds of Music Listened to by Weekdays")
    plt.xticks(rotation=45)
    plt.savefig("weekdays.png")


def make_chart_2(weekend_ser):
    day_ser = pd.Series()
    day_ser["Weekdays"] = "Weekdays"
    day_ser["Weekends"] = "Weekends"
    plt.figure()
    plt.bar(day_ser, weekend_ser, color="green")
    plt.xlabel("Week Time")
    plt.ylabel("Number of Seconds of Music Listened to")
    plt.title("Average Number of Seconds of Music Listened to by Weekdays or Weekends")
    plt.xticks(rotation=45)
    plt.savefig("weekends.png")

def make_sep_df(streaming_df):

    week_df = pd.DataFrame(columns=streaming_df.columns)
    condition = streaming_df.Weekdays == "Weekday"
    rows = streaming_df.loc[condition, :]
    week_df = week_df.append(rows, ignore_index=False)
    
    wend_df = pd.DataFrame(columns=streaming_df.columns)
    condition2 = streaming_df.Weekdays == "Weekend"
    rows2 = streaming_df.loc[condition2, :]
    wend_df = wend_df.append(rows2, ignore_index=False)

    return week_df, wend_df

def pool_by_day(streaming_df, name):
    days_ser = pd.DataFrame(index=streaming_df["msPlayed"])
    days_ser = streaming_df.groupby(["date"]).sum('sPlayed')
    weekday_ser = pd.Series(streaming_df["msPlayed"])
    days_ser = days_ser["sPlayed"]

    days_ser.to_csv(name)

    days_ser_2 = pd.read_csv("day.csv")
    date = days_ser_2["date"]
    new_date_ser = pd.Series()
    new_date_ser_2 = pd.Series()
    
    new_date_ser = pd.to_datetime(date)
    new_date_ser_2 = new_date_ser.dt.day_name()
    days_ser_2["Day_of_Week"] = new_date_ser_2

    for i in range(len(new_date_ser_2)):
        if new_date_ser_2[i] == "Monday" or new_date_ser_2[i] == "Tuesday" or new_date_ser_2[i] == "Wednesday" or new_date_ser_2[i] == "Thursday" or new_date_ser_2[i] == "Friday":
            weekday_ser[i] = "Weekday"
        elif  new_date_ser_2[i] == "Saturday" or new_date_ser_2[i] == "Sunday":
            weekday_ser[i] = "Weekend"
    days_ser_2["Weekdays"] = weekday_ser
    week_df, wend_df = make_sep_df(days_ser_2)
    return days_ser_2, week_df, wend_df

def make_chart_3(days_ser_2):
    day_ser = days_ser_2["date"]
    time_ser = days_ser_2["sPlayed"]
    plt.figure()
    plt.bar(day_ser, time_ser, color="green")
    plt.xlabel("Date between 9/17/2020 and 9/17/2021")
    plt.ylabel("Number of Seconds of Music Listened to")
    plt.title("Number of Seconds of Music Listened to by day")
    plt.xticks([])
    plt.savefig("days.png")

def run_t_test(week_df, wend_df):
    week_time_ser = week_df["sPlayed"]
    weekend_time_ser = wend_df["sPlayed"]
    print("H0: week = weekend"
    "Ha: week =/ weekend")

    print("We are going to run a 2-sampled, 2-tailed, t-test to see if ther is a difference between music playing times on weekends versus weekdays. I will use the significance level of .05 and df of infity due to there being >50000 cases. This means we will reject H0 if the critical value is > 1.96 or <-1.96 and fail to reject if it is in between.")

    t, p = stats.ttest_ind(week_time_ser, weekend_time_ser, equal_var=False)

    print("t:", t, "p:", p)

    print("Since t>1.96, we reject H0 at a significance level of .05. There appears to be a difference between weekday and weekend playing times")


def seperate_time_data(streaming_df):
    time_ser = streaming_df["endTime"]
    placeholder = []
    for i in range(len(time_ser)):
        placeholder.append(str(time_ser[i]))
    time_ser_2 = pd.Series(placeholder)
    new_df = time_ser_2.str.split(expand=True, pat=":")
    hour_ser = new_df[0]
    time_ser_3 = pd.Series(streaming_df["endTime"])
    for i in range(len(hour_ser)):
        if hour_ser[i] == "00" or hour_ser[i] == "01" or hour_ser[i] == "02" or hour_ser[i] == "03":
            time_ser_3[i] = 1
        if hour_ser[i] == "04" or hour_ser[i] == "05" or hour_ser[i] == "06" or hour_ser[i] == "07":
            time_ser_3[i] = 2
        if hour_ser[i] == "08" or hour_ser[i] == "09" or hour_ser[i] == "10" or hour_ser[i] == "11":
            time_ser_3[i] = 3
        if hour_ser[i] == "12" or hour_ser[i] == "13" or hour_ser[i] == "14" or hour_ser[i] == "15":
            time_ser_3[i] = 4
        if hour_ser[i] == "16" or hour_ser[i] == "17" or hour_ser[i] == "18" or hour_ser[i] == "19":
            time_ser_3[i] = 5
        if hour_ser[i] == "20" or hour_ser[i] == "21" or hour_ser[i] == "22" or hour_ser[i] == "23":
            time_ser_3[i] = 6
    streaming_df["time_index"] = time_ser_3
    
    return streaming_df


def split_time(streaming_df):
    one_df = pd.DataFrame(columns=streaming_df.columns)
    condition = streaming_df.time_index == 1
    rows = streaming_df.loc[condition, :]
    one_df = one_df.append(rows, ignore_index=False)
    
    two_df = pd.DataFrame(columns=streaming_df.columns)
    condition = streaming_df.time_index == 2
    rows = streaming_df.loc[condition, :]
    two_df = two_df.append(rows, ignore_index=False)

    three_df = pd.DataFrame(columns=streaming_df.columns)
    condition = streaming_df.time_index == 3
    rows = streaming_df.loc[condition, :]
    three_df = three_df.append(rows, ignore_index=False)

    four_df = pd.DataFrame(columns=streaming_df.columns)
    condition = streaming_df.time_index == 4
    rows = streaming_df.loc[condition, :]
    four_df = four_df.append(rows, ignore_index=False)

    five_df = pd.DataFrame(columns=streaming_df.columns)
    condition = streaming_df.time_index == 5
    rows = streaming_df.loc[condition, :]
    five_df = five_df.append(rows, ignore_index=False)

    six_df = pd.DataFrame(columns=streaming_df.columns)
    condition = streaming_df.time_index == 6
    rows = streaming_df.loc[condition, :]
    six_df = six_df.append(rows, ignore_index=False)
    
    
    print(one_df)
    one_df = pool_by_day(one_df, "one.csv")
    two_df = pool_by_day(two_df, "two.csv")
    three_df = pool_by_day(three_df, "three.csv")
    four_df = pool_by_day(four_df, "four.csv")
    five_df = pool_by_day(five_df, "five.csv")
    six_df = pool_by_day(six_df, "six.csv")
    print(one_df)
    one_df = pd.DataFrame(one_df)
    two_df = pd.DataFrame(two_df)
    three_df = pd.DataFrame(three_df)
    four_df = pd.DataFrame(four_df)
    five_df = pd.DataFrame(five_df)
    six_df = pd.DataFrame(six_df)
    print(one_df)
    one_ser = one_df["sPlayed"]
    two_ser = two_df["sPlayed"]
    three_ser = three_df["sPlayed"]
    four_ser = four_df["sPlayed"]
    five_ser = five_df["sPlayed"]
    six_ser = six_df["sPlayed"]
    time_index = streaming_df["time_index"]
    s_ser = streaming_df["sPlayed"]
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    six = 0
    for i in range(len(time_index)):
        if time_index[i] == 1:
            one += s_ser[i]
        if time_index[i] == 2:
            two += s_ser[i]
        if time_index[i] == 3:
            three += s_ser[i]
        if time_index[i] == 4:
            four += s_ser[i]
        if time_index[i] == 5:
            five += s_ser[i]
        if time_index[i] == 6:
            six += s_ser[i]    
    
    Day_ser_2 = pd.Series()
    Day_ser_2["12am to 4am"] = "12am to 4am"
    Day_ser_2["4am to 8am"] = "4am to 8am"
    Day_ser_2["8am to 12pm"] = "8am to 12pm"
    Day_ser_2["12pm to 4pm"] = "12pm to 4pm"
    Day_ser_2["4pm to 8pm"] = "4pm to 8pm"
    Day_ser_2["8pm to 12am"] = "8pm to 12am"

    new_time_ser = pd.Series()
    new_time_ser["12am to 4am"] = three
    new_time_ser["4am to 8am"] = four
    new_time_ser["8am to 12pm"] = five
    new_time_ser["12pm to 4pm"] = six
    new_time_ser["4pm to 8pm"] = one
    new_time_ser["8pm to 12pm"] = two
    return one_ser, two_ser, three_ser, four_ser, five_ser, six_ser, new_time_ser, Day_ser_2

def make_chart_4(new_time_ser, day_ser_2):
    plt.figure()
    plt.bar(day_ser_2, new_time_ser, color="blue")
    plt.xlabel("Time")
    plt.ylabel("Number of Seconds of Music Listened to")
    plt.title("Number of Seconds of Music Listened to by Time")
    plt.xticks(rotation=45)
    plt.savefig("times.png")

