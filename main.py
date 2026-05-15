#import
import cv2
import numpy as np
import os
import shutil
import subprocess
from skimage.morphology import skeletonize

#config
datasetImgWidth = 300
datasetImgHeight = 300
debug = True

#function
def separateImg(fileName, i):
    inputImgWidth = float(input("Enter the width of the input image %d (meter): "%(i+1)))
    inputImgHeight = float(input("Enter the height of the input image %d (meter): "%(i+1)))

    if((inputImgWidth % datasetImgWidth) <= (datasetImgWidth/2)):
        numImgWidth = int(inputImgWidth / datasetImgWidth)
    else:
        numImgWidth = int(inputImgWidth / datasetImgWidth) + 1

    if((inputImgHeight % datasetImgHeight) <= (datasetImgHeight/2)):
        numImgHeight = int(inputImgHeight / datasetImgHeight)
    else:
        numImgHeight = int(inputImgHeight / datasetImgHeight) + 1

    img = cv2.imread(f"./input/{fileName}")
    size = img.shape
    xsize = [0,0]
    xsize[0] = int(size[0] / numImgHeight)
    xsize[1] = int(size[1] / numImgWidth)

    for w in range(numImgWidth):
        for h in range(numImgHeight):
            ltop = (w*xsize[1], h*xsize[0])
            rtbm = ((w+1)*xsize[1], (h+1)*xsize[0])
            img_cap = img[ltop[1]:rtbm[1], ltop[0]: rtbm[0]]
            cv2.imwrite(f"./src/test_images/{fileName[0:-4]}_{ltop[0]}_{ltop[1]}.png", img_cap)

    return [rtbm[0], rtbm[1]]

def combineImg(fileName, finalImgSize):
    list = os.listdir("./mgTemp")
    img_list = []
    for file in list:
        if file.startswith(fileName[0:-4]):
            img_list.append(file)
    
    final_img = np.zeros((int(finalImgSize[1]), int(finalImgSize[0]), 3), dtype=np.uint8)
    for img in img_list:
        w = int(img.split("_")[-2])
        h = int(img.split("_")[-1].split(".")[0])
        img_cap = cv2.imread(f"./mgTemp/{img}")
        size = img_cap.shape
        final_img[h:h+size[0], w:w+size[1]] = img_cap
        os.remove(f"./mgTemp/{img}")

    cv2.imwrite(f"./mgTemp/{fileName[0:-4]}.png", final_img)

def separateObj(dirName, color, objName):
    mask = cv2.imread(f"./mgTemp/{dirName}")

    mask2 = cv2.inRange(mask, color, color)

    cv2.imwrite(f"./mgTemp/{dirName[0:-4]}_{objName}.png", mask2)
    if (debug):
        cv2.imwrite(f"./test/{dirName[0:-4]}_{objName}.png", mask2)

def combineObj(dirName):
    road = cv2.imread(f"./mgTemp/{dirName[0:-4]}_road.png", 0)
    building = cv2.imread(f"./mgTemp/{dirName[0:-4]}_building.png", 0)
    water = cv2.imread(f"./mgTemp/{dirName[0:-4]}_water.png", 0)
    tree = cv2.imread(f"./mgTemp/{dirName[0:-4]}_tree.png", 0)

    h, w = road.shape

    final_map = np.zeros((h,w,3), dtype=np.uint8)

    final_map[:] = (255, 255, 255)

    final_map[water > 0] = (184,163,50)

    final_map[tree > 0] = (0,200,0)

    final_map[building > 0] = (200,200,200)

    final_map[road > 0] = (100,100,100)

    cv2.imwrite(f"./output/{dirName}", final_map)

def contourSmoothing(dirName,objName):
    mask = cv2.imread(f"./mgTemp/{dirName[0:-4]}_{objName}.png", 0)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for cnt in contours:

        epsilon = 0.0005 * cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(
            cnt,
            epsilon,
            True
        )

        cv2.drawContours(
            result,
            [approx],
            -1,
            (255,255,255),
            -1
        )

    cv2.imwrite(f"./mgTemp/{dirName[0:-4]}_{objName}.png", result)
    return(f"./mgTemp/{dirName[0:-4]}_{objName}.png")

def contourPolygonization(dirName, objName):

    building = cv2.imread(f"./mgTemp/{dirName[0:-4]}_{objName}.png", 0)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(building)

    result = np.zeros((building.shape[0], building.shape[1], 3), dtype=np.uint8)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        # 過濾太小雜訊
        if area < 20:
            continue

        component = np.uint8(labels == i) * 255

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            epsilon = 0.015 * cv2.arcLength(cnt, True)

            approx = cv2.approxPolyDP(cnt, epsilon, True)

            cv2.drawContours(
                result,
                [approx],
                -1,
                (255,255,255),
                -1
            )

    cv2.imwrite(
        f"./mgTemp/{dirName[0:-4]}_{objName}.png",
        result
    )
    return(f"./mgTemp/{dirName[0:-4]}_{objName}.png")

def skeletonization(dirName, objName):

    road = cv2.imread(f"./mgTemp/{dirName[0:-4]}_{objName}.png", 0)

    kernel = np.ones((8,8), np.uint8)

    road = cv2.morphologyEx(
        road,
        cv2.MORPH_OPEN,
        kernel
    )

    road = cv2.GaussianBlur(road, (5,5), 0)

    _, road = cv2.threshold(
        road,
        127,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((30,30), np.uint8)

    road = cv2.morphologyEx(
        road,
        cv2.MORPH_CLOSE,
        kernel
    )

    road_bool = road > 0

    skeleton = skeletonize(road_bool)

    skeleton = (skeleton * 255).astype(np.uint8)

    kernel = np.ones((12,12), np.uint8)
    thick_road = cv2.dilate(skeleton, kernel, iterations=1)

    # kernel = np.ones((25,25), np.uint8)

    # thick_road = cv2.morphologyEx(
    #     thick_road,
    #     cv2.MORPH_CLOSE,
    #     kernel
    # )

    cv2.imwrite(
        f"./mgTemp/{dirName[0:-4]}_{objName}.png",
        thick_road
    )

    return(f"./mgTemp/{dirName[0:-4]}_{objName}.png")

#main
def main():
    try:
        shutil.rmtree("./mgTemp")
    except:
        pass
    os.makedirs("./mgTemp", exist_ok=True)
    try:
        shutil.rmtree("./src/test_images")
    except:
        pass
    os.makedirs("./src/test_images", exist_ok=True)

    list = os.listdir("./input")
    finalImgSize = np.zeros((len(list), 2), dtype=np.float32)

    for i in range(len(list)):
        finalImgSize[i] = separateImg(list[i],i)

    subprocess.run(["python", "src/predict.py", "--dataset", "oem", "--input", "src/test_images", "--ckpt", "src/checkpoints/best_deeplabv3plus_mobilenet_oem_os16.pth", "--save_val_results_to", "mgTemp"], shell=True)

    for i in range(len(list)):
        combineImg(list[i], finalImgSize[i])

    list = os.listdir("./mgTemp")
    for dirName in list:
        separateObj(dirName, np.array([3, 1, 14]), "water")
        separateObj(dirName, np.array([234, 41, 131]), "tree")
        separateObj(dirName, np.array([215, 171, 57]), "building")
        separateObj(dirName, np.array([27, 219, 129]), "road")

        contourSmoothing(dirName, "water")
        contourSmoothing(dirName, "tree")
        contourPolygonization(dirName, "building")
        skeletonization(dirName, "road")

        combineObj(dirName)

    if(not debug):
        shutil.rmtree("./mgTemp")

if __name__ == "__main__":
    main()