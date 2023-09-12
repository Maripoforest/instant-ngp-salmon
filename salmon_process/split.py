import os
import shutil

original_dir = './flame_salmon_1/4fps'
new_dir = './salmon/frames/'

os.makedirs(new_dir, exist_ok=True)

# for camera_index in range(18):

frame_number = 1
while frame_number < 159:
    frame_dir = os.path.join(new_dir, f'{frame_number:04d}')
    os.makedirs(frame_dir, exist_ok=True)  
    for camera_index in range(21):
        if camera_index == 3 or camera_index == 17:
            continue
        camera_dir = os.path.join(original_dir, f'cam{camera_index:02d}')
        original_image_path = os.path.join(camera_dir, f'{frame_number:04d}.jpg')
        new_image_path = os.path.join(frame_dir, f'{camera_index:04d}.jpg')
        shutil.copy(original_image_path, new_image_path)
    frame_number += 1
