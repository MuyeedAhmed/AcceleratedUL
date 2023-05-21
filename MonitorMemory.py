import psutil
import time

def monitor_memory_usage(interval):
    print("Memory usage")
    while True:
        memory = []
        for _ in range(10):
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.used / (1024 * 1024)  # Convert to megabytes
            memory.append(memory_usage)
            time.sleep(interval)
        print([ '%.2f' % elem for elem in memory])
        

# Example usage
monitor_memory_usage(interval=1)
