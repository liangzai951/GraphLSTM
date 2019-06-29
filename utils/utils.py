from cv2 import cv2
import numpy as np
from skimage import img_as_float
from skimage.segmentation import slic

from config import N_SUPERPIXELS, IMAGE_SHAPE


def average_rgb_for_superpixels(image, segments):
    averages = []
    for segment_value in np.unique(segments):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        mask[segments == segment_value] = 255
        non_zero_pixels_amount = np.count_nonzero(mask)
        filtered = cv2.bitwise_and(image, image, mask=mask)
        av_local = []
        for c in range(filtered.shape[-1]):
            av_local.append(np.sum(filtered[:, :, c], axis=1).sum() / non_zero_pixels_amount)
        i = av_local.index(max(av_local))
        av_local = [0.0] * len(av_local)
        av_local[i] = 1.0
        averages.append(av_local)
    while len(averages) != N_SUPERPIXELS:
        averages.append([0.0] * IMAGE_SHAPE[-1])
    return averages


def obtain_superpixels(image, n_segments, sigma):
    return slic(img_as_float(image), n_segments=n_segments, sigma=sigma)


def get_neighbors(segments, n_segments):
    # get unique labels
    vertices = np.unique(segments)
    reverse_dict = dict(zip(vertices, np.arange(len(vertices))))
    segments = np.array([reverse_dict[x] for x in segments.flat]).reshape(
        segments.shape)
    down = np.c_[segments[:-1, :].ravel(), segments[1:, :].ravel()]
    right = np.c_[segments[:, :-1].ravel(), segments[:, 1:].ravel()]
    all_edges = np.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges, axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:, 0] + num_vertices * all_edges[:, 1]
    edges = [[vertices[x % num_vertices], vertices[x // num_vertices]] for x in
             np.unique(edge_hash)]
    e = {v: set([x[1] for x in edges if x[0] == v] + [x[0] for x in edges if
                                                      x[1] == v]) for v in
         sorted(vertices)}
    matrix = np.zeros(shape=(n_segments, n_segments))
    for start_node, neighbors in e.items():
        for neighbor in neighbors:
            matrix[start_node, neighbor] = 1
            matrix[neighbor, start_node] = 1
    return matrix


def sort_values(values: list, neighbors, start_node=0, mode="bfs"):
    assert len(values) == len(neighbors)
    start_vertex = start_node
    mapping = []
    visited, queue = set(), [start_vertex]
    result = []
    if mode == "bfs":
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                mapping.append(vertex)
                result.append(values[vertex])
                local_neighbors = set([i for i, value in enumerate(neighbors[vertex].tolist()) if value == 1.0])
                queue.extend(local_neighbors - visited)
    elif mode == "dfs":
        while queue:
            vertex = queue.pop()
            if vertex not in visited:
                visited.add(vertex)
                mapping.append(vertex)
                result.append(values[vertex])
                local_neighbors = set([i for i, value in enumerate(neighbors[vertex].tolist()) if value == 1.0])
                queue.extend(local_neighbors - visited)
    while len(result) != N_SUPERPIXELS:
        result.append([0.0] * IMAGE_SHAPE[-1])
        mapping.append(len(result)-1)
    return result, mapping, np.argsort(mapping).tolist()


def get_superpixels_index_for_hot_areas(slic_matrix):
    height, width = slic_matrix.shape
    height -= 1
    width -= 1
    north = slic_matrix[0, int(np.floor(width / 2))]
    south = slic_matrix[height, int(np.floor(width / 2))]
    west = slic_matrix[int(np.floor(height / 2)), 0]
    east = slic_matrix[int(np.floor(height / 2)), width]
    center = slic_matrix[int(np.floor(height / 2)), int(np.floor(width / 2))]
    north_west = slic_matrix[0, 0]
    north_east = slic_matrix[0, width]
    south_west = slic_matrix[height, 0]
    south_east = slic_matrix[height, width]
    return center, north, east, south, west, north_east, south_east, south_west, north_west