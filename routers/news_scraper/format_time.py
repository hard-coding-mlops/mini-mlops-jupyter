from datetime import datetime, timedelta
import pytz

def get_formatted_current_date_time():
    return datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y.%m.%d. %H:%M:%S")

def get_formatted_current_date():
    return datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d")

def format_date(date_time):
    parsed_date = datetime.strptime(date_time, "%Y. %m. %d. %H:%M")
    formatted_date = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_date