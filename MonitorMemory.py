import psutil
import time

def monitor_memory_usage(interval):
    print("Memory usage")
    while True:
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.used / (1024 * 1024)  # Convert to megabytes
        print(f"{memory_usage:.2f} MB", end=' ')
        time.sleep(interval)

# Example usage
monitor_memory_usage(interval=1)
