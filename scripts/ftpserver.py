from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
import logging
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation 
import cv2
import json
from datetime import datetime
import time
import os
from glob import glob
import socket
import shutil

def quaternion_to_matrix(q):
    x, y, z, w = q['x'], q['y'], q['z'], q['w']
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return [
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy), 0],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx), 0],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy), 0],
        [                0,                 0,                 0, 1]
    ]

def opencv_to_colmap_pose(R, t):
    R = np.asarray(R)
    t = np.asarray(t).reshape(3, 1)
    t[2] = -t[2]
    R_colmap = R.T
    
    return R_colmap, t

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def transform_tfs(json_data):
    up = np.zeros(3)
    for frame in json_data['frames']:
        transform_matrix = np.array(frame['transform_matrix'], dtype=float)
        
        R_opencv = transform_matrix[:3, :3]
        # rotation_z = Rotation.from_euler('z', -90, degrees=True)
        # R_opencv_new = rotation_z.apply(R_opencv)
        # transform_matrix[:3, :3] = R_opencv_new

        # t_opencv = transform_matrix[:3, 3]

        c2w = transform_matrix
        c2w[:3,:3] = R_opencv
        c2w[0:3,2] *= -1 # flip the y and z axis
        c2w[0:3,1] *= -1
        c2w = c2w[[1,0,2,3],:]
        c2w[2,:] *= -1 # flip whole world upside down
        up += c2w[0:3,1]
        # c2w[:3, 3] *= 2
        
        # R_colmap, t_colmap = opencv_to_colmap_pose(R_opencv, t_opencv)
        # new_transform_matrix = np.eye(4)
        # new_transform_matrix[:3, :3] = R_colmap
        # new_transform_matrix[:3, 3] = t_colmap.ravel()
        
        frame['transform_matrix'] = c2w.tolist()

    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1

    for f in json_data["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in json_data["frames"]:
        mf = f["transform_matrix"][0:3,:]
        for g in json_data["frames"]:
            mg = g["transform_matrix"][0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.00001:
                totp += p*w
                totw += w
    if totw > 0.0:
        totp /= totw
    print(totp) # the cameras are looking at totp
    for f in json_data["frames"]:
        f["transform_matrix"][0:3,3] -= totp

    avglen = 0.
    for f in json_data["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
    avglen /= len(json_data["frames"])
    print("avg camera distance from origin", avglen)
    for f in json_data["frames"]:
        f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"
    
    for f in json_data["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    return json_data

def save_transforms(transforms, output_file):
    with open(output_file, 'w') as f:
        json.dump(transforms, f, indent=4)

def convert():
    scene = 'newscene'
    base_path = 'data/nerf/'
    folder_path = base_path + scene + '/raw/all'  # Change this to your folder path
    output_file = base_path + scene + '/raw/unrotated.json'
    camera_params = {
        "camera_angle_x": 2 * np.arctan(1280 / (2 * 909.6491088867188)),  # Updated for the new intrinsics
        "camera_angle_y": 2 * np.arctan(720 / (2 * 909.9080200195312)),   # Updated for the new intrinsics
        "fl_x": 909.6491088867188,
        "fl_y": 909.9080200195312,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": 634.2947998046875,
        "cy": 379.4480895996094,
        "w": 1280.0,
        "h": 720.0,
        "aabb_scale": 4,
        "frames": []
    }

    json_files = sorted(glob(os.path.join(folder_path, '*.json')))
    image_index = 1

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        image_file = os.path.join(folder_path, f"{base_name}.png")
        file_path = f"images/{base_name}.png"  # Assuming you will convert PNGs to JPGs or change accordingly
        
        translation = data['translation']
        rotation = data['rotation']
        
        transform_matrix = quaternion_to_matrix(rotation)
        transform_matrix[0][3] = translation['x']
        transform_matrix[1][3] = translation['y']
        transform_matrix[2][3] = translation['z']

        frame = {
            "file_path": file_path,
            "sharpness": 30.0,  # Placeholder value, replace with actual sharpness calculation if available
            "transform_matrix": transform_matrix
        }

        camera_params["frames"].append(frame)
        image_index += 1

    with open(output_file, 'w') as f:
        json.dump(camera_params, f, indent=4)

def json2nerf():
    scene = 'newscene'
    base_path = 'data/nerf/'
    input_file = base_path + scene + '/raw/unrotated.json' 
    with open(input_file, 'r') as f:
        transforms_data = json.load(f)
    
    transformed_data = transform_tfs(transforms_data)
    
    output_file = base_path + scene + '/transforms.json' 
    save_transforms(transformed_data, output_file)
    
    print(f"Transformed tfs saved to {output_file}")

def clear_directory_contents(directory_path):
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory and its contents
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"Directory {directory_path} does not exist.")

# Set up logging
logging.basicConfig(filename='ftp_server.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomFTPHandler(FTPHandler):
    def on_file_received(self, file_path):
        # Check if the file is an image based on its extension
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        _, ext = os.path.splitext(file_path.lower())

        if ext in image_extensions:
            logging.info(f"Image file received: {file_path}")
            
            if self.count >= 2:
                convert()  
                json2nerf() 
            else:
                self.count += 1
        else:
            logging.debug(f"Non-image file received: {file_path}")

def run_ftp_server():
    clear_directory_contents('data/nerf/newscene/images')
    clear_directory_contents('data/nerf/newscene/raw/all')

    authorizer = DummyAuthorizer()
    authorizer.add_user("user", "12345", ".", perm="elradfmw")

    handler = CustomFTPHandler
    handler.authorizer = authorizer
    handler.count = 0

    # Set the server to listen on IP address 192.168.4.3 and port 2121
    server = FTPServer(("192.168.4.3", 2121), handler)

    print("FTP server running on port 2121...")
    server.serve_forever()

if __name__ == "__main__":
    run_ftp_server()
