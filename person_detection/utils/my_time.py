import time
import datetime
from my_requests import get_ping_time
import requests
import sched

# time.strftime("%Y-%m-%d %H:%M:%S")
# time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def get_timestamp(Y_m_d_H_M_S):
    # "%Y-%m-%d %H:%M:%S"
    return time.mktime(time.strptime(Y_m_d_H_M_S, "%Y-%m-%d %H:%M:%S"))

def timestamp_to_datetime(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)

def datetime_to_timestamp(_datetime):
    return time.mktime(_datetime.timetuple())

def timedelta(**args):
    d = datetime.datetime.now() + datetime.timedelta(**args)
    return datetime_to_timestamp(d)

def sched_to_run_demo():
    def run(name):
        print(name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # 设置时间调度器
    s = sched.scheduler(time.time, time.sleep)
    # 设置运行时间 enterabs(时间， 优先级， 调用的函数， 参数)
    now = time.time()
    s.enterabs(now + 2, 1, run, ('first',))
    s.enterabs(now + 2, 0, run, ('second',))
    # 运行函数
    s.run()

def get_taobao_timestamp():
    url = 'http://api.m.taobao.com/rest/api3.do?api=mtop.common.getTimestamp'
    resp = requests.get(url=url)
    net = int(resp.json()['data']['t']) * 1000 # microseconds
    elapsed = resp.elapsed.microseconds
    ping_time = int(get_ping_time('api.m.taobao.com') * 1e6)
    local = int(time.time()*1e6)
    print("淘宝时间戳: ", net)
    print("包请求时间: ", elapsed)
    print("ping time: ", ping_time)
    print("服务器处理请求时间: ", elapsed - ping_time)
    print("本地时间戳: ", local)
    print(local - net)

if __name__ == '__main__':
    # print(get_timestamp('2016-11-24 14:00:21'))
    # t = timedelta(days=-10)
    # print(timestamp_to_datetime(t))
    # get_taobao_timestamp()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    pass