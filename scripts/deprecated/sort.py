import os

# directory = "./snapshots"

# for i in range(1, 177):
#     old_file = os.path.join(directory, f"{i}.msgpack")
#     new_file = os.path.join(directory, f"{i-1}.msgpack")
#     if os.path.exists(old_file):
#         os.rename(old_file, new_file)
#     else:
#         print(f"{i} not found")


directory = "./video0010"

for i in range(1, 177):
    old_file = os.path.join(directory, f"{i}.png")
    new_file = os.path.join(directory, f"{i-1}.png")
    if os.path.exists(old_file):
        os.rename(old_file, new_file)
    else:
        print(f"{i} not found")