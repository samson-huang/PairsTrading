from datetime import datetime
import time
import os
from apscheduler.schedulers.background import BackgroundScheduler

def tick():
    print('Tick! The time is: %s' % datetime.now())

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    # 间隔3秒钟执行一次
    scheduler.add_job(tick, 'interval', seconds=3)
    # 这里的调度任务是独立的一个线程
    scheduler.start()
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))
    try:
        # 其他任务是独立的线程执行
        while True:
            time.sleep(2)
            print('sleep!')
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print('Exit The Job!')