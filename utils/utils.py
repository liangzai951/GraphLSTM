import operator
import random

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
        averages.append(
            [np.sum(filtered[:, :, c], axis=1).sum() / non_zero_pixels_amount
             for c in
             range(filtered.shape[-1])])
    while len(averages) != N_SUPERPIXELS:
        averages.append([0.0] * IMAGE_SHAPE[-1])
    return averages


def obtain_superpixels(image, n_segments, sigma):
    return slic(img_as_float(image), n_segments=n_segments, sigma=sigma)


def get_confidence_map(image, segments):
    return {i: random.uniform(a=0, b=1) for i in np.unique(segments)}


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


def sort_values(values: dict, neighbors: dict, confidence_map: dict,
                mode="dfs"):
    assert len(confidence_map) == len(values) == len(neighbors)
    if mode == "confidence":
        sorted_vertices = sorted(confidence_map.items(),
                                 key=operator.itemgetter(1), reverse=True)
        sorted_vertices = [x[0] for x in sorted_vertices]
        return [values[v] for v in sorted_vertices]
    else:
        start_vertex = \
        sorted(confidence_map.items(), key=operator.itemgetter(1),
               reverse=True)[0][0]
        visited, queue = set(), [start_vertex]
        result = []
        if mode == "bfs":
            while queue:
                vertex = queue.pop(0)
                if vertex not in visited:
                    visited.add(vertex)
                    result.append(values[vertex])
                    queue.extend(neighbors[vertex] - visited)
        elif mode == "dfs":
            while queue:
                vertex = queue.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    result.append(values[vertex])
                    queue.extend(neighbors[vertex] - visited)
        return result
