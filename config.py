MODEL_PATH = "glstm.hdf5"
# MODEL_PATH = "./data/checkpoints/model_01_0.13.hdf5"

IMAGE_SHAPE = (500, 500, 3)
SLIC_SHAPE = (IMAGE_SHAPE[0], IMAGE_SHAPE[1])
N_SUPERPIXELS = 1000
N_FEATURES = 5

SLIC_SIGMA = 5

TRAINSET_FILE = "./data/dataset/VOC2012/ImageSets/Segmentation/train.txt"
TRAINVALSET_FILE = "./data/dataset/VOC2012/ImageSets/Segmentation/trainval.txt"
VALSET_FILE = "./data/dataset/VOC2012/ImageSets/Segmentation/val.txt"

IMAGES_PATH = "./data/dataset/VOC2012/JPEGImages/"
VALIDATION_IMAGES = "./data/dataset/VOC2012/SegmentationObject/"

OUTPUT_PATH = "./data/output/"


TRAIN_ELEMS = 1464
TRAIN_BATCH_SIZE = 2 * 48

VALIDATION_ELEMS = 2913
VALIDATION_BATCH_SIZE = 1 * 48
