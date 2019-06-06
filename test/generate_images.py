import random

from PIL import Image, ImageDraw

from config import IMAGE_SHAPE

COLOR_DICT = {0: (255, 0, 0, 255), 1: (0, 255, 0, 255), 2: (0, 0, 255, 255)}

if __name__ == '__main__':
    for i in range(33):
        img = Image.new('RGB', IMAGE_SHAPE[:2])
        dr = ImageDraw.Draw(img)
        dr.rectangle(((0, 0), IMAGE_SHAPE[:2]), fill=COLOR_DICT[random.randint(0, len(COLOR_DICT)-1)])
        for j in range(random.randint(2, 8)):
            x1 = random.randint(5, IMAGE_SHAPE[1]-15)
            x2 = x1 + random.randint(10, IMAGE_SHAPE[1]-5) + 5
            y1 = random.randint(5, IMAGE_SHAPE[0]-15)
            y2 = y1 + random.randint(10, IMAGE_SHAPE[0]-5) + 5

            x1 = max(min(x1, IMAGE_SHAPE[1]-5), 5)
            x2 = max(min(x2, IMAGE_SHAPE[1]-5), 5)
            y1 = max(min(y1, IMAGE_SHAPE[0]-5), 5)
            y2 = max(min(y2, IMAGE_SHAPE[0]-5), 5)

            fill = COLOR_DICT[random.randint(0, len(COLOR_DICT)-1)]

            dr.rectangle(((x1, y1), (x2, y2)), fill=fill)
        img.save("../data/test/test_{0!s}.png".format(i))
