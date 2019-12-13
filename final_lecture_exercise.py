import autodiff.autodiff as ad
import time
import numpy as np
from matplotlib import pyplot as plt


nums = [10, 100, 1000, 10000, 100000]
times = []
for num in nums:
    #num = 100
    inputs = []
    start = time.time()
    for i in range(num):
        der = np.zeros(num)
        der[i] = 1.
        inputs.append(ad.AutoDiff(1, der))
    
    func = lambda x: np.prod(x)
    output = func(inputs)
    times.append(time.time() - start)
    print(times[-1])

fig, ax = plt.subplots()
ax.plot(nums, times)
plt.show()
