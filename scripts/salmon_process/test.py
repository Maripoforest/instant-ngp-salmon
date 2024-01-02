import json

new_name = "transforms_ngp.json"
with open("./flame_salmon_1/transforms.json", "r") as f:
    transforms = json.load(f)
tf_list = list()

print(len(transforms["cams"]))


new_tf = dict()
new_tf["camera_angle_x"] = 1.4942718116546532
new_tf["camera_angle_y"] = 1.213802434248789
new_tf["fl_x"] = 1459.6319139996715
new_tf["fl_y"] = 1460.42945664873
new_tf["k1"] = 0.003725608120667334
new_tf["k2"] = 0.0006838527629254057
new_tf["k3"] = 0.0
new_tf["k4"] = 0.0    
new_tf["p1"] = 0.0002806999541777068
new_tf["p2"] = 0.0009806766015162491
new_tf["cx"] = 1356.964130859679
new_tf["cy"] = 1014.3325859769806
new_tf["w"] = 2704.0
new_tf["h"] = 2028.0
new_tf["aabb_scale"] = 32
new_tf["frames"] = list()
for i in range(len(transforms["cams"])):
    frame = dict()
    frame["file_path"] = "images/" + str(i).zfill(4) + ".jpg"
    frame["sharpness"] = 61.0
    frame["transform_matrix"] = transforms["cams"][i]
    new_tf["frames"].append(frame)

# dump the new_tf as json
with open(new_name, "w") as f:
    json.dump(new_tf, f, indent=4)
