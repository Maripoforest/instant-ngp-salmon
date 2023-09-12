import numpy as np
import json

file_path = './poses_bounds.npy'  # Replace with the path to your .npy file
data = np.load(file_path)
print(data)
pose = data[:, :-2].reshape([-1, 3, 5])
t_pose = pose.transpose([1, 2, 0])
transform = []
_add = np.array([0, 0, 0, 1])
print(pose.shape)
for slice in pose:
    _slice = slice[:,:-1]
    _slice = np.vstack((_slice, _add))
    transform.append(_slice)

transform = np.array(transform)
transform = transform.tolist()

json_file_path = 'transforms.json'

data_js = dict()
data_js["cams"] = []
for row in transform:
    data_js['cams'].append(row)

with open(json_file_path, 'w') as json_file:
    json.dump(data_js, json_file)

