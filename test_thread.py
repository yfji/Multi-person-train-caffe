# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 16:01:19 2018

@author: samsung
"""

import threading
import numpy as np
from collections import deque
import time

maxQueueLen=1
num_gpus=2
stopFlag=False
time_thresh=2
LOOP=10
datum_id=0
Data_size=200
time_delay=0.2

class Datum(object):
    def __init__(self):
        self.data=None
        self.n_ID=0
        self.empty=True
        
    def feedData(self, data, n_id):
        self.data=data
        self.n_ID=n_id
        self.empty=False
    
    def getData(self):
        return self.data
    
    def getID(self):
        return self.n_ID
        
    def copyTo(self, datum):
        datum.data=self.data.copy()
        datum.n_ID=self.n_ID
        datum.empty=self.empty

class Queue(object):
    def __init__(self, name='', queueLen=0):
        self.que=deque(maxlen=queueLen)
        self.lock=threading.Lock()
        self.pushFlag=True
        self.popFlag=True
        self.name=name
        
    def tryPush(self, datum):
        try:
            self.lock.acquire()
            if len(self.que)==maxQueueLen:
                return False
            else:
                self.que.append(datum)
#                print('Queue %s try push success'%self.name)
                return True
        except:
            return False
        finally:
            self.lock.release() #executed before all returns
    
    def tryPop(self):
        try:
            self.lock.acquire()
            if len(self.que)==0:
                return Datum()
            else:
                datum=self.que.popleft()
#                print('Queue %s try pop success'%self.name)
                return datum
        except:
            return None
        finally:
            self.lock.release()
    
    def forcePush(self, datum):
        try:
            self.lock.acquire()
            self.que.append(datum)
#            print('Queue %s force push success'%self.name)
            return True
        except:
            return False
        finally:
            self.lock.release()
    
    def waitAndPush(self, datum):
        try:
            start=time.time()
            while self.pushFlag:
                self.lock.acquire()
                if len(self.que)<maxQueueLen:
                    break
                self.lock.release()
                time.sleep(1e-3)
                end=time.time()
                if end-start>=time_thresh:
                    return False
            if not self.pushFlag:
                return False
            self.que.append(datum)
#            print('Queue %s wait push success'%self.name)
            return True
        except:
            return False
        finally:
            if self.lock.locked():
                self.lock.release()
                
    def waitAndPop(self):
        try:
            start=time.time()
            while self.popFlag:
                self.lock.acquire()
                if len(self.que)>0:
                    break
                self.lock.release()
                time.sleep(1e-3)
                end=time.time()
                if end-start>=time_thresh:
                    return None
            if not self.popFlag:
                return None
            datum=self.que.popleft()
#            print('Queue %s wait pop success'%self.name)
            return datum
        except:
            return None
        finally:
            if self.lock.locked():
                self.lock.release()
    
    def push(self, datum):
        return self.waitAndPush(datum)
        
    def pop(self, datum):
        return self.waitAndPop()
        
    def stopQueue(self):
        self.pushFlag=False
        self.popFlag=False
        self.que.clear()

class ThreadManager(object):
    def __init__(self, threads):
        self.gpuThreads=threads
        self.extras=[]
        
    def start(self):
        for t in self.extras:
            t.start()
        for i in range(num_gpus):
            for t in self.gpuThreads[i]:
                t.start()
        for t in self.extras:
            t.join()
        for i in range(num_gpus):
            for t in self.gpuThreads[i]:
                t.join()
                
    def addExtras(self, *args):
        for t in args:
            self.extras.append(t)
            
    def stop(self):
        for t in self.extras:
            t.stopAndJoin()
        for i in range(num_gpus):
            for t in threads:
                t.stopAndJoin()

class Thread(threading.Thread):
    def __init__(self, thread_id, sub_threads):
        self.threadId=thread_id
        self.subThreads=sub_threads
        self.thread=None
        self.loop=0
        self.isRunning=True
        self.isRunningSuccess=True
        self.thread=threading.Thread(target=self.threadFunction)
        
    def threadFunction(self):
        global stopFlag
        while self.isRunning and not stopFlag:
            for t in self.subThreads:
                self.isRunningSuccess&=t.work()
#            if self.isRunningSuccess:
#                self.loop+=1
#        if self.loop==LOOP:
#            stopFlag=True

    def start(self):
        self.thread.start()
        
    def stop(self):
        self.isRunning=False
        
    def join(self):
        self.thread.join()
        
    def stopAndJoin(self):
        self.stop()
        self.join()
        
class SubThread(object):
    def __init__(self, workers):
        self.workers=workers
    
    def workTWorkers(self, datum):
        for worker in self.workers:
            worker.work(datum)
            
    def work(self, datum):
        pass
            
class SubThreadOut(SubThread):
    def __init__(self, queueOut, workers):
        super(SubThreadOut, self).__init__(workers)
        self.queueOut=queueOut
    
    def work(self):
        datum=Datum()
        self.workTWorkers(datum)
        if not self.queueOut.waitAndPush(datum):
            print('Queue %s Time out'%self.queueOut.name)
            return False
        else:
            return True
            
        
class SubThreadIn(SubThread):
    def __init__(self, queueIn, workers):
        super(SubThreadIn, self).__init__(workers)
        self.queueIn=queueIn
        
    def work(self):
        datum=self.queueIn.tryPop()
        if datum is None:
            print('Queue %s raise exception'%self.queueIn.name)
            return False
        elif datum.empty:
#            print('Queue %s empty'%self.queueIn.name)
            return False
        else:
#            print('Consumer work datum')
            self.workTWorkers(datum)
            return True
        
class SubThreadInOut(SubThread):
    def __init__(self, queueIn, queueOut, workers):
        super(SubThreadInOut, self).__init__(workers)
        self.queueIn=queueIn
        self.queueOut=queueOut
        
    def work(self):
        datum=self.queueIn.tryPop()
        if datum is None:
            print('Queue %s raise exception'%self.queueIn.name)
            return False
        elif datum.empty:
#            print('Queue %s empty'%self.queueIn.name)
            return False
        else:
            self.workTWorkers(datum)
            if not self.queueOut.waitAndPush(datum):
                print('Queue %s Time out'%self.queueOut.name)
                return False
            else:
#                print('Transferer work datum')
                return True
    
class Worker(object):
    def __init__(self):
        pass
    
    def work(self, datum):
        pass
    
class Producer(Worker):
    def __init__(self):
        super(Producer, self).__init__()
    
    def workProducer(self):
        global datum_id
        datum=Datum()
        datum.feedData(np.ones((10,10), dtype=np.float32), datum_id)
        datum_id+=1
        return datum
        
    def work(self, datum):
        d=self.workProducer()
        d.copyTo(datum)
            
class Transferer(Worker):
    def __init__(self):
        super(Transferer, self).__init__()
    
    def workTransferer(self, datum):
        datum.data*=2.0
        
    def work(self, datum):
        self.workTransferer(datum)
        time.sleep(time_delay)

class Consumer(Worker):
    def __init__(self):
        super(Consumer, self).__init__()
        self.dataPool=[]
        self.poolsize=0
    
    def workConsumer(self, datum):
        global Data_size, stopFlag
        self.dataPool.append(datum)
        if len(self.dataPool)==self.poolsize:
            sorted(self.dataPool, key=lambda x:x.n_ID)
            for d in self.dataPool:
                if d.n_ID==Data_size-1:
                    stopFlag=True
            d=self.dataPool[len(self.dataPool)-1]
            print('Datum info (id: h,w,mean,id): (%d,%d,%d,%f)'%(d.n_ID, d.data.shape[0],d.data.shape[1],d.data.mean()))
            self.dataPool=[]
        
    def work(self, datum):
        self.workConsumer(datum)

class KeyThread(threading.Thread):
    def __init__(self):
        super(KeyThread,self).__init__()
        
    def run(self):
        pass
    
if __name__=='__main__':
    print('Starting openpose thread demo...Press ESC to exit')
    gpu_ids=[0]
    threads=[]
    
    publicQueueIn=Queue(name='queueIn', queueLen=num_gpus*maxQueueLen)
    publicQueueOut=Queue(name='queueOut', queueLen=num_gpus*maxQueueLen)
    
    
    producer=Producer()
    consumer=Consumer()
    consumer.poolsize=num_gpus*maxQueueLen
    
    sub_prod=SubThreadOut(publicQueueOut, [producer])
    sub_cons=SubThreadIn(publicQueueIn, [consumer])
    thread_prod=Thread(0, [sub_prod])
    thread_cons=Thread(2, [sub_cons])
    
    all_threads=[]
    for i in range(num_gpus):
        transferer=Transferer()        
        sub_trans=SubThreadInOut(publicQueueOut, publicQueueIn, [transferer])
        thread_trans=Thread(1+i*num_gpus, [sub_trans])
        m_threads=[thread_trans]
        all_threads.append(m_threads)
    
    manager=ThreadManager(all_threads)
    manager.addExtras(thread_prod, thread_cons)
    
    start=time.time()
    manager.start()
    duration=time.time()-start
        
    print('Finish using %f seconds on %d GPUs'%(duration, num_gpus))
