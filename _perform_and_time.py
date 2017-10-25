import time

def perform_and_time(message, fun, *args):
    t0 = time.time()
    rtn = fun(*args)
    t1 = time.time()
    result = ' (s): %g' % (t1-t0)
    print(''.join((message, result)))
    return rtn

