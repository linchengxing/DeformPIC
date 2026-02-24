import math
import os
import random
import pickle
import numpy as np
import json
import tqdm

def random_rotate(pointcloud, level=1):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    angle_clip = math.pi / 3
    angle_clip = angle_clip / 3 * level
    angles = np.random.uniform(-angle_clip, angle_clip, size=(3))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    pointcloud = np.dot(pointcloud, R)
    return pointcloud, R

def y_flip(pointcloud):
    angles = [0, 0, math.pi]
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    pointcloud = np.dot(pointcloud, R)
    return pointcloud

def random_dropout_global(pointcloud, level=1):
    """
    Drop random points globally
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    drop_rate = [0.5, 0.75, 0.875, 0.9375, 0.96875][level - 1]
    num_points = pointcloud.shape[0]
    pointcloud[(1 - int(drop_rate * num_points)):, :] = 0
    return pointcloud

def random_add_noise(pointcloud, level=1, sigma=0.2):
    """
    Randomly add noise data to point cloud
    :param pointcloud: input point cloud
    :param num_noise: number of noise points
    :return: corrupted point cloud
    """
    N, _ = pointcloud.shape
    num_noise = 100 * level
    noise = np.clip(sigma * np.random.randn(num_noise, 3), -1, 1)
    idx = np.random.randint(0, N, num_noise)
    pointcloud[idx, :3] = pointcloud[idx, :3] + noise
    return pointcloud

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def main():
    tasks = ["registration", "denoising", "reconstruction"]
    root_path = "data/ModelNet40"
    test_output_dir = "data/ModelNet40/Test_dataset"
    json_output_dir = "data/ModelNet40"

    with open(os.path.join(root_path, "modelnet40_normal_resampled", "modelnet40_shape_names.txt")) as f:
        shape_names = [line.rstrip() for line in f]

    new_test_list = []

    test_data_path = os.path.join(root_path, f"modelnet40_test_8192pts_fps.dat")
    with open(test_data_path, 'rb') as f:
        test_list_of_points, test_list_of_labels = pickle.load(f)
    list_of_points = test_list_of_points
    list_of_labels = test_list_of_labels
    for task in tasks:
        level_dict = {}
        combined_list = list(zip(list_of_points, list_of_labels))
        random.shuffle(combined_list)
        points_list, labels_list = zip(*combined_list)
        for idx, point_set in tqdm.tqdm(enumerate(points_list), total=len(points_list)):
            point_set = farthest_point_sample(point_set[:, :3], 1024)  # (1024, 3)
            point_set = pc_normalize(point_set)  # (-1, 1)

            category = labels_list[idx].item()
            category_name = shape_names[category]
            file_name = "{}_{}.npy".format(category_name, idx)

            length = len(points_list)
            if idx < length / 5:
                level = 1
            elif length / 5 <= idx < length / 5 * 2:
                level = 2
            elif length / 5 * 2 <= idx < length / 5 * 3:
                level = 3
            elif length / 5 * 3 <= idx < length / 5 * 4:
                level = 4
            else:
                level = 5

            if level not in level_dict:
                level_dict[level] = [idx]
            else:
                level_dict[level].append(idx)

            source_output_path = os.path.join(test_output_dir, task, "sources/level{}".format(level), file_name)
            if task == "reconstruction":
                target = point_set.copy()
                target = random_dropout_global(target, level=level)
            elif task == "denoising":
                target = point_set.copy()
                target = random_add_noise(target, level=level)
            elif task == "registration":
                target = point_set.copy()
                point_set = y_flip(point_set)
                target, R = random_rotate(target, level=level)
            else:
                raise NotImplementedError()
            target_output_path = os.path.join(test_output_dir, task, "targets/level{}".format(level), file_name)

            if os.path.exists(source_output_path):
                print(source_output_path)

            if not os.path.exists(os.path.join(test_output_dir, task, "sources/level{}".format(level))):
                os.makedirs(os.path.join(test_output_dir, task, "sources/level{}".format(level)))
            if not os.path.exists(os.path.join(test_output_dir, task, "targets/level{}".format(level))):
                os.makedirs(os.path.join(test_output_dir, task, "targets/level{}".format(level)))

            if task == "registration":
                np.savez(source_output_path[:-4], pointcloud=target, rotation_matrix=R)
                str = os.path.join("Test_dataset", task, "sources/level{}".format(level), file_name[:-1] + 'z') # .npz
            else:
                np.save(source_output_path, target)
                str = os.path.join("Test_dataset", task, "sources/level{}".format(level), file_name) # .npy
            np.save(target_output_path, point_set)

            new_test_list.append(str)
        print("----------------------{} process successfully!!!--------------------".format(task))
        print("Total: {} samples".format(len(points_list)))

        print("Including")
        for i in range(5):
            print("---------level{}: {}".format(i + 1, len(level_dict[i + 1])))

    json.dump(new_test_list, open(os.path.join(json_output_dir, "test_list.json"), 'w'))


if __name__ == "__main__":
    main()