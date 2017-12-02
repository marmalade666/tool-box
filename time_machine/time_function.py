import datetime,os,sys
__author__ = 'sladesal'
__time__ = '20171202'

def test_print():
	print('this is a test example!')

def timefun(sched_timer):
	flag = 1
	while  True:
		now = datetime.datetime.now()
		if now == sched_timer:
			test_print()
			flag=1
		else:
			if flag == 1:
				sched_timer = sched_timer + datetime.timedelta(minutes=1)
				print('the next sched_timer is %s'%sched_timer)
				flag = 0

if __name__ == '__main__':
	sched_timer = datetime.datetime(2017,11,17,17,47,50)
  	pyfile = sys.argv[1]
  	python pyfile
	print('run the code at %s' %sched_timer) 
	timefun(sched_timer)
