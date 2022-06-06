import os
import time
total_time = 0.0
reps_done = 0
for i in range(10):
    if i < 5: 
       os.system("./test.sh")
       continue
    start = time.time()
    os.system("./test.sh")
    end = time.time()
    delta = end - start
    total_time += delta
    reps_done += 1
avg_time = total_time / reps_done
latency = avg_time * 1000
throughput = 1.0 / avg_time
print("Throughput: {:.2f} fps".format(throughput))