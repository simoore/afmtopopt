class PrintObject(object):
    def __init__(self, name):
        self.name = name
        
    def to_console(self):
        print(self.name)


class FuncExecutor(object):
    def __init__(self, func):
       self._func = func
       
    def execute(self, obj):
        self._func(obj)
        FuncExecutor.execute2()
        
    def execute2():
        print('how')
    
    
o1 = PrintObject('one')
o2 = PrintObject('two')
f1 = FuncExecutor(PrintObject.to_console)
f1.execute(o1)
f1.execute(o2)