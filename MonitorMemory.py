import psutil
import time
import numpy as np

def monitor_memory_usage(interval):
    print("Memory usage")
    f=open("Test/Time.csv", "w")
    f.write('Time,Memory\n')
    f.close()

    while True:
        memory = []
        for _ in range(10):
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.used / (1024 * 1024)  # Convert to megabytes
            memory.append(memory_usage)
            time.sleep(interval)
        # print([ '%.2f' % elem for elem in memory])
        
        f=open("Test/Time.csv", "a")
        f.write(str(time.time())+","+str(np.mean(memory))+'\n')
        f.close()



def monitor_memory_usage_pid(pid, interval):
    print("Memory usage")
    
    f=open("Test/Time.csv", "w")
    f.write('Name,Time,Memory_Physical,Memory_Virtual\n')
    f.close()
    
    while True:
        memory = []
        memory_virtual = []
        name, pid = get_max_pid()
        for _ in range(10):
            try:
                process = psutil.Process(pid)
                memory_info = process.memory_info()
                memory_usage = (memory_info.rss) / (1024 * 1024)  # Convert to megabytes
                memory.append(memory_usage)
                memory_usage_virtual = memory_info.vms / (1024 * 1024)
                memory_virtual.append(memory_usage_virtual)
            except:
                print("none")
                continue
            time.sleep(interval)
        # print(name, pid)
        # print([ '%.2f' % elem for elem in memory])
        # print([ '%.2f' % elem for elem in memory_virtual])
        
        f=open("Test/Time.csv", "a")
        f.write(name+","+str(time.time())+","+str(np.mean(memory))+","+str(np.mean(memory_virtual))+'\n')
        f.close()
        
def get_max_pid():
    processes = psutil.process_iter(['pid', 'name', 'memory_info'])
    # Initialize variables to track the highest memory usage
    max_memory = 0
    max_memory_pid = None
    
    # Iterate over the processes
    for process in processes:
        # Get the memory usage information for each process
        try:
            memory_info = process.info['memory_info']
            memory_usage = memory_info.rss
        except:
            memory_usage = 0
        # Check if the current process has higher memory usage
        if memory_usage > max_memory and 'python' in process.info['name'].lower():
            max_memory = memory_usage
            max_memory_name = process.info['name']
            max_memory_pid = process.info['pid']
    return max_memory_name, max_memory_pid


monitor_memory_usage_pid(38511, interval=1)


# monitor_memory_usage(interval=1)