from argparse import ArgumentParser, RawTextHelpFormatter
from skimage import io, exposure
import matplotlib.pyplot as plt
import numpy as np


# to [0,255]
def toUINT8(grayImage):
    grayImage[grayImage < 0] = 0
    grayImage[grayImage > 255] = 255
    return grayImage.astype(np.uint8, copy=False)


def getHistogram(grayImage):
    histogram = np.zeros(256, dtype=np.float32)
    m, n = grayImage.shape
    for i in range(m):
        for j in range(n):
            histogram[grayImage[i, j]] += 1.0
    return histogram


def getNormalizedHistogram(grayImage):
    histogram = getHistogram(grayImage)
    pixelCount = grayImage.shape[0] * grayImage.shape[1]
    return histogram / pixelCount

def getNormalizedLevelHistogram(grayImage, qmin, qmax):
    histogram = np.zeros(256, dtype=np.float32)
    m, n = grayImage.shape
    pixelCount = 0
    for i in range(m):
        for j in range(n):
            if qmin <= grayImage[i,j] < qmax:
                histogram[grayImage[i, j]] += 1.0
                pixelCount += 1
    return histogram / pixelCount


def getAccum(nHistogram):
    accum = np.zeros_like(nHistogram)
    accum[0] = nHistogram[0]
    for i in range(1, len(nHistogram)):
        accum[i] = accum[i - 1] + nHistogram[i]
    return accum


def equalize(grayImage, qmin, qmax):
    accum = getAccum(getNormalizedHistogram(grayImage))
    equalized = np.zeros_like(grayImage)
    m, n = grayImage.shape
    for i in range(m):
        for j in range(n):
            if qmin <= grayImage[i, j] < qmax:
                equalized[i, j] = (accum[grayImage[i, j]] * (qmax - qmin)) + qmin
    return equalized


# def createLevelSet(grayImage, iters):
#     splits = 2 ** iters
#     print("iteracion:", iters, "levelSets:", splits)
#     splitIntervals = list(range(0, 257, round(256 / splits)))
#     levelSets = []
#     for i in range(splits):
#         qmin = splitIntervals[i]
#         qmax = splitIntervals[i + 1]
#         print("levelset:", i, "qmin:", qmin, "qmax:", qmax)
#         print(grayImage[100:101, 100:101])
#         level = equalize(grayImage, qmin, qmax)
#         io.imshow(level)
#         io.show()
#         print("EQ:")
#         print(level[100:102, 100:102])
#         levelSets.append(level)
#     return levelSets



def improve2(grayImage, iters):

    # Por cada iter: Crear el level set, ajustar los contrastes y unir las imagenes.
    for i in range(iters + 1):
        splits = 2 ** i
        splitIntervals = list(range(0, 257, round(256 / splits)))

        print("iteracion:", i, "levelSets:", splits, "intervals:", splitIntervals)

        if i == 0:
            nHistogram = getNormalizedHistogram(grayImage)
            accum = getAccum(nHistogram)

            figure = plt.figure()
            f1 = figure.add_subplot(131)
            f2 = figure.add_subplot(132)
            f3 = figure.add_subplot(133)

            f1.imshow(grayImage, cmap="gray")
            f2.bar(left=np.arange(256), height=nHistogram)
            f3.bar(left=np.arange(256), height=accum)

        improved = np.zeros_like(grayImage)

        # Para cada rango, mejorar las imagenes:
        for j in range(splits):

            # qmin = float(256)*i/2**j;
            # qmax = float(256)*i/2**(j+1)

            qmin = splitIntervals[j]
            qmax = splitIntervals[j + 1]

            print("levelset:", j, "qmin:", qmin, "qmax:", qmax)

            # accum = getAccum(getNormalizedLevelHistogram(grayImage, qmin, qmax))

            equalized = np.zeros_like(grayImage)
            m, n = grayImage.shape

            for r in range(m):
                for p in range(n):
                    if qmin <= grayImage[r, p] < qmax:
                        equalized[r, p] = (accum[grayImage[r, p]] * (qmax - qmin)) + qmin
                    # else: 0

            improved = improved + equalized

        print("merge:", i)
        grayImage = improved

        nHistogram = getNormalizedHistogram(grayImage)
        accum = getAccum(nHistogram)

        figure = plt.figure()
        f1 = figure.add_subplot(131)
        f2 = figure.add_subplot(132)
        f3 = figure.add_subplot(133)

        f1.imshow(grayImage, cmap="gray")
        f2.bar(left=np.arange(256), height=nHistogram)
        f3.bar(left=np.arange(256), height=accum)

    plt.show()

    return grayImage


# def improve(grayImage, iters):
#     levelSetUnion = grayImage
#     for i in range(iters + 1):
#         levelSet = createLevelSet(levelSetUnion, i)
#         levelSetUnion = np.zeros_like(grayImage)
#         for j in range(len(levelSet)):
#             levelSetUnion += levelSet[j]
#     return levelSetUnion


def main():
    parser = ArgumentParser(prog="LevelSets",
                            formatter_class=RawTextHelpFormatter,
                            description="Mejoramiento de imagenes por Level Sets\n"
                                        "\n\nCodificar text.txt en image.jpg usando 2 LSB será:\n"
                                        ">> python tarea_1.py --encode --image image.jpg --text text.txt --nbits 2"
                                        "\n\nPara decodificar será:\n"
                                        ">>python tarea_1.py --decode --image imagen.jpg"
                            )
    parser.add_argument('-i', '--input',
                        dest='imageFilename',
                        action='store',
                        help='Nombre del archivo a mejorar')

    parser.add_argument('-t', '--t',
                        dest='iterNums',
                        action='store',
                        help='Numero de iteraciones para LevelSets')

    args = parser.parse_args()
    print(args)

    grayImage = toUINT8(255 * io.imread(args.imageFilename, as_grey=True))
    improved = improve2(grayImage, 3)

    # fig = plt.figure()
    #
    # print("Sin iterar")
    # x11 = fig.add_subplot(5, 2, 1)
    # x11.imshow(grayImage, cmap='gray', vmin=0, vmax=255)
    # x11.set_title('Original')
    # x11.axis('off')
    #
    # x12 = fig.add_subplot(5, 2, 2)
    # x12.bar(left=np.arange(256), height=getNormalizedHistogram(grayImage))
    # x12.axis('off')
    #
    # print("improve: 0")
    # improved = improve2(grayImage, 0)
    #
    # x13 = fig.add_subplot(5, 2, 3)
    # x13.imshow(improved, cmap='gray', vmin=0, vmax=255)
    # x13.axis('off')
    #
    # x14 = fig.add_subplot(5, 2, 4)
    # x14.bar(left=np.arange(256), height=getNormalizedHistogram(improved))
    # x14.axis('off')
    #
    # print("improve: 1")
    # improved = improve2(grayImage, 1)
    #
    # x15 = fig.add_subplot(5, 2, 5)
    # x15.imshow(improved, cmap='gray', vmin=0, vmax=255)
    # x15.axis('off')
    #
    # x16 = fig.add_subplot(5, 2, 6)
    # x16.bar(left=np.arange(256), height=getNormalizedHistogram(improved))
    # x16.axis('off')
    #
    # print("improve: 2")
    # improved = improve2(grayImage, 2)
    #
    # x17 = fig.add_subplot(5, 2,7)
    # x17.imshow(improved, cmap='gray', vmin=0, vmax=255)
    # x17.axis('off')
    #
    # x18 = fig.add_subplot(5, 2, 8)
    # x18.bar(left=np.arange(256), height=getNormalizedHistogram(improved))
    # x18.axis('off')
    #
    # print("improve: 3")
    # improved = improve2(grayImage, 3)
    #
    # x17 = fig.add_subplot(5, 2, 9)
    # x17.imshow(improved, cmap='gray', vmin=0, vmax=255)
    # x17.axis('off')
    #
    # x18 = fig.add_subplot(5, 2, 10)
    # x18.bar(left=np.arange(256), height=getNormalizedHistogram(improved))
    # x18.axis('off')
    #
    # plt.show()

    # improve(image, args.iterNums)


if __name__ == "__main__":
    main()
