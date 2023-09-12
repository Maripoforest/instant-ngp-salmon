import numpy as np
import json

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next is not None:
            current = current.next
        current.next = new_node

def list_to_linked_list(input_list):
    linked_list = LinkedList()
    for item in input_list:
        linked_list.append(item)
    return linked_list


if __name__ == "__main__":

    with open("./transforms.json", 'r') as json_file:
        data_dict = json.load(json_file)

    campose_list = list_to_linked_list(data_dict["cams"])
    p = campose_list.head
    
    height = 2028
    width = 2704
    focal = 1458.49997

    frames = list()
    for i in range(10):
        j = 0
        serial_number = f"{i+1:04}"
        p = campose_list.head
        while j < 10:
            if j == 3:
                j += 1
            frame = dict()
            folder = "cam" + f"{j:02}" + "/"   
            
            frame["file_path"] = "./salmon/" + folder + serial_number + ".jpg"
            frame["rotation"] = 0.79
            frame["time"] = 40 / 160 * i + 0.05
            # frame["transform_matrix"] = p.data
            llff_m = np.array(p.data)
            
            theta = np.radians(90)
            _rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            rotation_matrix = llff_m[:3, :3]
            rotation_matrix = np.dot(rotation_matrix, _rotation_matrix)
            llff_m[:3, :3] = rotation_matrix

            frame["transform_matrix"] = llff_m.tolist()
            p = p.next
            j += 1
            frames.append(frame)


    transforms_train = dict()
    transforms_train["camera_angle_x"] = 2 * np.arctan(width / (2 * focal)) 
    transforms_train["frames"] = frames

    with open("./transforms_train.json", 'w') as json_file:
        json.dump(transforms_train, json_file)
    with open("./transforms_val.json", 'w') as json_file:
        json.dump(transforms_train, json_file)




