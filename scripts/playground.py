# import threading
# from pipeline import NeRFPipeline

# ppl = NeRFPipeline()
# ppl.delay = True
# pipeline_thread = threading.Thread(target=ppl.spinning)
# pipeline_thread.start()

import numpy as np
import matplotlib.pyplot as plt
import json

with open("./result.json", "r") as f:
    delay = json.load(f)
with open("./result_no.json", "r") as f:
    no_delay = json.load(f)

plt.plot(delay, label="delay")
plt.plot(no_delay, label="no delay")
plt.legend()
plt.show()

# for i in range(50):
#     if i < 10:

#         continue
#     print("continue")
#     print(i)