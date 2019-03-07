from skimage.segmentation import mark_boundaries
from skimage import io
import matplotlib.pyplot as plt

from .utils import obtain_superpixels, get_confidence_map, get_neighbors, average_rgb_for_superpixels, sort_values

if __name__ == '__main__':
    image = io.imread("lena.png")
    segments = obtain_superpixels(image, 10, 5)
    print(segments)
    print(segments.shape)
    print(type(segments))
    confidence_map = get_confidence_map(image, segments)
    neighbors = get_neighbors(segments, 10)
    superpixels_vectors = average_rgb_for_superpixels(image, segments)
    print(confidence_map)
    print(neighbors)
    print(superpixels_vectors)
    # print("==================================")
    # print(sort_values(superpixels_vectors, neighbors, confidence_map, "confidence"))
    # print("==================================")
    # print(sort_values(superpixels_vectors, neighbors, confidence_map, "bfs"))
    # print("==================================")
    # print(sort_values(superpixels_vectors, neighbors, confidence_map, "dfs"))

    fig = plt.figure("Superpixels")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")
    plt.show()
