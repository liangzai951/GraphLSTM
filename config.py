import keras.backend as K

MODEL_PATH = "glstm.hdf5"
EPOCHS = 400
RAW_MODEL_PATH = "glstm_raw.hdf5"
VALIDATION_MODEL = "../data/checkpoints/model_340_0.15_0.13_0.83_0.85.hdf5"
IMAGE_SHAPE = (250, 250, 3)
SLIC_SHAPE = (IMAGE_SHAPE[0], IMAGE_SHAPE[1])
N_SUPERPIXELS = 100
N_FEATURES = 3
INPUT_PATHS = 2

SLIC_SIGMA = 0

TRAINSET_FILE = "./data/dataset/VOC2012/ImageSets/Segmentation/train.txt"
TRAINVALSET_FILE = "./data/dataset/VOC2012/ImageSets/Segmentation/trainval.txt"
VALSET_FILE = "./data/dataset/VOC2012/ImageSets/Segmentation/val.txt"

IMAGES_PATH = "./data/dataset/VOC2012/JPEGImages/"
VALIDATION_IMAGES = "./data/dataset/VOC2012/SegmentationClass/"

OUTPUT_PATH = "./"


TRAIN_ELEMS = 23
TRAIN_BATCH_SIZE = VALIDATION_BATCH_SIZE = PREDICT_BATCH_SIZE = 1

VALIDATION_ELEMS = 10
image_list = ["test_{0!s}".format(i) for i in range(TRAIN_ELEMS + VALIDATION_ELEMS)]

def custom_mse(y_true, y_pred): return K.mean(K.square(y_pred - y_true), axis=1)
