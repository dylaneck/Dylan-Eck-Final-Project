
from numpy.lib.shape_base import expand_dims
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


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
    #print(time_ser)

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
    print(streaming_df)
    week_df = pd.DataFrame(columns=streaming_df.columns)
    condition = streaming_df.Weekdays == "Weekday"
    rows = streaming_df.loc[condition, :]
    week_df = week_df.append(rows, ignore_index=False)
    
    wend_df = pd.DataFrame(columns=streaming_df.columns)
    condition2 = streaming_df.Weekdays == "Weekend"
    rows2 = streaming_df.loc[condition2, :]
    wend_df = wend_df.append(rows2, ignore_index=False)
    
    monday_df = pd.DataFrame(columns=streaming_df.columns)
    condition2 = streaming_df.Day_of_Week == "Monday"
    rows2 = streaming_df.loc[condition2, :]
    monday_df = monday_df.append(rows2, ignore_index=False)
    
    tuesday_df = pd.DataFrame(columns=streaming_df.columns)
    condition2 = streaming_df.Day_of_Week == "Tuesday"
    rows2 = streaming_df.loc[condition2, :]
    tuesday_df = tuesday_df.append(rows2, ignore_index=False)
    
    wednesday_df = pd.DataFrame(columns=streaming_df.columns)
    condition2 = streaming_df.Day_of_Week == "Wednesday"
    rows2 = streaming_df.loc[condition2, :]
    wednesday_df = wednesday_df.append(rows2, ignore_index=False)

    thursday_df = pd.DataFrame(columns=streaming_df.columns)
    condition2 = streaming_df.Day_of_Week == "Thursday"
    rows2 = streaming_df.loc[condition2, :]
    thursday_df = thursday_df.append(rows2, ignore_index=False)

    friday_df = pd.DataFrame(columns=streaming_df.columns)
    condition2 = streaming_df.Day_of_Week == "Friday"
    rows2 = streaming_df.loc[condition2, :]
    friday_df = friday_df.append(rows2, ignore_index=False)

    saturday_df = pd.DataFrame(columns=streaming_df.columns)
    condition2 = streaming_df.Day_of_Week == "Saturday"
    rows2 = streaming_df.loc[condition2, :]
    saturday_df = saturday_df.append(rows2, ignore_index=False)

    sunday_df = pd.DataFrame(columns=streaming_df.columns)
    condition2 = streaming_df.Day_of_Week == "Sunday"
    rows2 = streaming_df.loc[condition2, :]
    sunday_df = sunday_df.append(rows2, ignore_index=False)
    return week_df, wend_df, monday_df, tuesday_df, wednesday_df, thursday_df, friday_df, saturday_df, sunday_df

def pool_by_day(streaming_df, name):
    days_ser = pd.DataFrame()
    days_ser = streaming_df.groupby(["date"]).sum('sPlayed')
    weekday_ser = pd.Series(streaming_df["msPlayed"])
    #days_ser = days_ser["sPlayed"]

    days_ser.to_csv(name)

    days_ser_2 = pd.read_csv(name)
    date = days_ser_2["date"]
    new_date_ser = pd.DataFrame()
    new_date_ser_2 = pd.DataFrame()
    
    new_date_ser = pd.to_datetime(date)
    new_date_ser_2 = new_date_ser.dt.day_name()
    days_ser_2["Day_of_Week"] = new_date_ser_2

    for i in range(len(new_date_ser_2)):
        if new_date_ser_2[i] == "Monday" or new_date_ser_2[i] == "Tuesday" or new_date_ser_2[i] == "Wednesday" or new_date_ser_2[i] == "Thursday" or new_date_ser_2[i] == "Friday":
            weekday_ser[i] = "Weekday"
        elif  new_date_ser_2[i] == "Saturday" or new_date_ser_2[i] == "Sunday":
            weekday_ser[i] = "Weekend"
    days_ser_2["Weekdays"] = weekday_ser

    return days_ser_2

def make_chart_3(days_ser_2):
    day_ser = days_ser_2["date"]
    time_ser = days_ser_2["sPlayed"]
    plt.figure()
    plt.bar(day_ser, time_ser, color="green")
    plt.xlabel("Date between 9/17/2020 and 9/15/2021")
    plt.ylabel("Number of Seconds of Music Listened to")
    plt.title("Number of Seconds of Music Listened to by day")
    plt.xticks([])
    plt.savefig("days.png")

def seperate_time_data(streaming_df):
    time_ser = streaming_df["endTime"]
    placeholder = []
    for i in range(len(time_ser)):
        placeholder.append(str(time_ser[i]))
    time_ser_2 = pd.Series(placeholder)
    new_df = time_ser_2.str.split(expand=True, pat=":")
    hour_ser = new_df[0]
    time_ser_3 = []
    streaming_df["endTime"] = time_ser_2
    for i in range(len(hour_ser)):
        if hour_ser[i] == "00" or hour_ser[i] == "01" or hour_ser[i] == "02" or hour_ser[i] == "03":
            time_ser_3.append(1)
        if hour_ser[i] == "04" or hour_ser[i] == "05" or hour_ser[i] == "06" or hour_ser[i] == "07":
            time_ser_3.append(2)
        if hour_ser[i] == "08" or hour_ser[i] == "09" or hour_ser[i] == "10" or hour_ser[i] == "11":
            time_ser_3.append(3)
        if hour_ser[i] == "12" or hour_ser[i] == "13" or hour_ser[i] == "14" or hour_ser[i] == "15":
            time_ser_3.append(4)
        if hour_ser[i] == "16" or hour_ser[i] == "17" or hour_ser[i] == "18" or hour_ser[i] == "19":
            time_ser_3.append(5)
        if hour_ser[i] == "20" or hour_ser[i] == "21" or hour_ser[i] == "22" or hour_ser[i] == "23":
            time_ser_3.append(6)
    time_ser_4  = pd.Series(time_ser_3)
    streaming_df["time_index"] = time_ser_4
    
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
    
    
   
    one_df = pool_by_day(one_df, "one.csv")
    two_df = pool_by_day(two_df, "two.csv")
    three_df = pool_by_day(three_df, "three.csv")
    four_df = pool_by_day(four_df, "four.csv")
    five_df = pool_by_day(five_df, "five.csv")
    six_df = pool_by_day(six_df, "six.csv")

    one_df = pd.DataFrame(one_df)
    two_df = pd.DataFrame(two_df)
    three_df = pd.DataFrame(three_df)
    four_df = pd.DataFrame(four_df)
    five_df = pd.DataFrame(five_df)
    six_df = pd.DataFrame(six_df)
    
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

def days_ttest(monday_df, tuesday_df, wednesday_df, friday_df, saturday_df, sunday_df):
    monday_ser = monday_df["sPlayed"]
    tuesday_ser = tuesday_df["sPlayed"]
    wednesday_ser = wednesday_df["sPlayed"]
    friday_ser = friday_df["sPlayed"]
    saturday_ser = saturday_df["sPlayed"]
    sunday_ser = sunday_df["sPlayed"]
    
    t1, p1 = stats.ttest_rel(tuesday_ser, saturday_ser)
    
    t2, p2  = stats.ttest_rel(monday_ser, friday_ser)

    t3, p3 = stats.ttest_rel(wednesday_ser, sunday_ser)

    print("t for Tuesday vs Saturday:", t1, "p:", p1)
    print("t for Monday vs Friday:", t2, "p:", p2)
    print("t for Wednesday vs Sunday:", t3, "p:", p3)

def weekday_ttest(week_df, wend_df):
    week_ser = week_df["sPlayed"]
    wend_ser = wend_df["sPlayed"]
    
    t1, p1 = stats.ttest_ind(week_ser, wend_ser)

    print("t:", t1, "p:", p1)

def drop_rows(streaming_df):
    streaming_df = streaming_df.drop(columns=["trackName", "artistName"])

    return streaming_df

def time_ttest(one_ser, two_ser, three_ser, four_ser, five_ser, six_ser):

    t1, p1 = stats.ttest_ind(three_ser, six_ser)
    
    t2, p2  = stats.ttest_ind(four_ser, one_ser)

    t3, p3 = stats.ttest_ind(five_ser, two_ser)

    print("t for 12-4:", t1, "p:", p1)
    print("t for 4-8:", t2, "p:", p2)
    print("t for 8-12:", t3, "p:", p3)

def kNN_class(streaming_df, num):
    
    rest_df  = streaming_df.drop(columns=["Weekdays", "Day_of_Week", "date", "year"])
    weekday_ser = streaming_df["year"]
    x_train, x_test, y_train, y_test = train_test_split(rest_df, weekday_ser, test_size=0.25, random_state=0)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    X_train_normalized = scaler.transform(x_train)
    X_test_normalized = scaler.transform(x_test)

    knn_clf = KNeighborsClassifier(n_neighbors=num, metric="euclidean")
    knn_clf.fit(X_train_normalized, y_train)
    y_predicted = knn_clf.predict(X_test_normalized)

    accuracy = accuracy_score(y_test, y_predicted)
    print("Accuracy:", accuracy)

def seperate_dates(streaming_df):
    date_ser = streaming_df["date"]
    date_df = date_ser.str.split(expand=True, pat="-")
    year_ser = date_df[0]
    day_ser = date_df[2]
    month_ser = date_df[1]
    streaming_df["year"] = year_ser
    streaming_df["month"] = month_ser

    return streaming_df


def make_warm_cold(streaming_df):
    month_ser = streaming_df["month"]
    placeholder = []
    for i in range(len(month_ser)):
    
        if month_ser[i] == "04" or month_ser[i] == "05" or month_ser[i] == "06" or month_ser[i] == "07" or month_ser[i] == "08" or month_ser[i] == "09":
            placeholder.append(1)
        else:
            placeholder.append(0)
    heat_ser = pd.Series(placeholder)
    streaming_df["heat"] = heat_ser

    return streaming_df

def clean_months(streaming_df):
    month_ser = streaming_df["month"]
    placeholder = []
    for i in range(len(month_ser)):
        if month_ser[i] == "01":
            placeholder.append(1)
        elif month_ser[i] == "02":
            placeholder.append(2)
        elif month_ser[i] == "03":
            placeholder.append(3)
        elif month_ser[i] == "04":
            placeholder.append(4)
        elif month_ser[i] == "05":
            placeholder.append(5)
        elif month_ser[i] == "06":
            placeholder.append(6)
        elif month_ser[i] == "07":
            placeholder.append(7)
        elif month_ser[i] == "08":
            placeholder.append(8)
        elif month_ser[i] == "09":
            placeholder.append(9)
        elif month_ser[i] == "10":
            placeholder.append(10)
        elif month_ser[i] == "11":
            placeholder.append(11)
        elif month_ser[i] == "12":
            placeholder.append(12)

    new_month_ser = pd.Series(placeholder)
    streaming_df["month"] = new_month_ser

    return streaming_df

def make_matrix(streaming_df):
    rest_df  = streaming_df.drop(columns=["msPlayed", "Day_of_Week", "date"])
    corr_df = rest_df.corr()
    print(corr_df.style.background_gradient(cmap='bwr').set_precision(2))


def convert_numeric(streaming_df):
    days_ser = streaming_df["Day_of_Week"]
    year_ser = streaming_df["year"]
    weekday_ser = streaming_df["Weekdays"]
    placeholder = []
    placeholder1 = []
    placeholder2 = []
    for i in range(len(days_ser)):
        if days_ser[i] == "Monday":
            placeholder.append(1)
        elif days_ser[i] == "Tuesday":
            placeholder.append(2)
        elif days_ser[i] == "Wednesday":
            placeholder.append(3)
        elif days_ser[i] == "Thursday":
            placeholder.append(4)
        elif days_ser[i] == "Friday":
            placeholder.append(5)
        elif days_ser[i] == "Saturday":
            placeholder.append(6)
        elif days_ser[i] == "Sunday":
            placeholder.append(7)
    new_days_ser = pd.Series(placeholder)
    streaming_df["Day_of_Week"] = new_days_ser
    
    for j in range(len(year_ser)):
        if year_ser[j] == "2020":
            placeholder1.append(0)
        elif year_ser[j] == "2021":
            placeholder1.append(1)
    new_year_ser = pd.Series(placeholder1)
    streaming_df["year"] = new_year_ser

    for k in range(len(weekday_ser)):
        if weekday_ser[k] == "Weekday":
            placeholder2.append(0)
        elif weekday_ser[k] == "Weekend":
            placeholder2.append(1)
    new_week_ser = pd.Series(placeholder2)
    streaming_df["Weekdays"] = new_week_ser

    return streaming_df


def tree_classifier(streaming_df):
    clf = DecisionTreeClassifier(random_state=0,max_depth=3)  
    rest_df  = streaming_df.drop(columns=["Weekdays", "Day_of_Week", "date", "year"])
    weekday_ser = streaming_df["year"]
    x_train, x_test, y_train, y_test = train_test_split(rest_df, weekday_ser, test_size = 0.25, random_state = 0)
    #test_instance = x_test

    clf.fit(x_train, y_train)
    #X_test = test_instance
    y_predicted = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    print("Accuracy:", accuracy)
    plt.figure(figsize=(30, 30)) # Resize figure
    plot_tree(clf, feature_names=x_train.columns, class_names={1: "2020", 0: "2021"}, filled=True)
    plt.show()




def normalize_week_data(week_df, wend_df):
    week_day_ser = week_df["Day_of_Week"]
    week_date_ser  = week_df["date"]
    week_time_ser = week_df["sPlayed"]
    weekend_day_ser = wend_df["Day_of_Week"]
    weekend_date_ser  = wend_df["date"]
    weekend_time_ser = wend_df["sPlayed"]
    placeholder = []
    placeholder1 = []

    week_number = week_date_ser['date'].dt.week
    weekend_num = weekend_date_ser['date'].dt.week
    
    for i in range(len(week_number)):
        weekday = 0

        weekday = weekday / 5
        weekend = weekend / 2
        placeholder.append(weekday)
        placeholder1.append(weekend)
    weekday_ser = pd.Series(placeholder)
    weekend_ser = pd.Series(placeholder1)
    return weekday_ser, weekend_ser