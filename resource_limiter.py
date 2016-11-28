import os
import psutil

def memory_usage_MB():
  '''Returns in MB the current virtual memory size'''
  denom = 1e6
  proc = psutil.Process(os.getpid())
  mem = proc.memory_info().vms
  return mem / denom

def memory_usage_pct():
  '''Returns in percent the current data memory size'''
  proc = psutil.Process(os.getpid())
  mempct = proc.memory_percent(memtype = 'vms')
  return mempct

#Seems to be different units from what the mem usages return?
def limit_memory_as(soft_lim):
  '''Sets the (soft) limit to the process's virtual memory usage'''
  proc = psutil.Process(os.getpid())
  _, hard_lim = proc.rlimit(psutil.RLIMIT_AS)
  proc.rlimit(psutil.RLIMIT_AS, (soft_lim, hard_lim))
  return

def test1():
  import numpy as np
  from matplotlib import pyplot as plt

  mem_use = []
  mem_lim = []
  N = 220
  step = 20
  mem_mb = 1
  limit_memory_as(int(mem_mb*1e6))
  for i in range(1, N, step):
    while True:
      try:
        a = np.ones((100**2)*(i**2))
        mem_lim.append(mem_mb)
        break
      except MemoryError:
        mem_mb += 10
        limit_memory_as(int(mem_mb*1e6))

    mem_use.append(memory_usage_MB())

  fig, ax1 = plt.subplots()
  ax1.plot(range(1, N, step), mem_use, linewidth = 2, label = 'MB used',
           color = 'b')
  ax1.set_ylabel('mem usage (MB)', color = 'b')
  ax1.set_xlabel('$i$')
  
  ax2 = ax1.twinx()
  ax2.plot(range(1, N, step), mem_lim, linewidth = 2, label = 'MB limit',
           color = 'r')
  ax2.set_ylabel('mem limit (MB)', color = 'r')

  plt.title('Memory Usage')
  plt.show()

  #there is no apparent pattern with the memory usage and the memory limit...
  plt.plot(range(1, N, step), [float(x) / y for x, y in zip(mem_use, mem_lim)])
  plt.show()

if __name__ == '__main__':
  limit_memory_as(int(7000e6))
  test1()
