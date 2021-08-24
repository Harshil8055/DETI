# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:56:16 2018

@author: surbhi
"""
import numpy as np
import sys
import cv2 as cv
import pandas as pd
from PIL import Image

import pytesseract as py
import csv



# def custombinary(img):
#    width, height = img.shape[:2]
#    for i in range(width - 1):
#        for j in range(height - 1):
#            if (img[i, j] < 127):
#                img[i, j] = 0
#            else:
#                img[i, j] = 255
#    return 0


def show_wait_destroy(winname, img):
    cv.imshow(winname,img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)


def main(argv):
    # [load_image]
    # Check number of arguments
    if len(argv) < 1:
        print('Not enough parameters')
        print('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1

    # Load the image
    src = cv.imread("table3.png")
    # cv.resize(src,(362,620))

    # Check if image is loaded fine
    if src is None:
        print('Error opening image: ' + argv[0])
        return -1

    # Show source image
    # cv.imshow("src", src)
    show_wait_destroy("src", src)

    cv.blur(src, (2, 2))
    # [load_image]
    # [gray]
    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src

    # Show gray image
    show_wait_destroy("gray", gray)
    # [gray]

    # [bin]
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv.bitwise_not(gray)
    show_wait_destroy("", gray)
    _, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Show binary image
    show_wait_destroy("binary", bw)

    # [bin]

    # [init]
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    # [init]

    # [horiz]
    # Specify size on horizontal axis
    cols = horizontal.shape[1]

    horizontal_size = int(cols / 8)

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    # print(horizontalStructure.shape)
    # show_wait_destroy("test",horizontal[5:6,0:118])

    #
    #    show_wait_destroy("test",horizontal[5:6,2:120])
    #    print(horizontal[5:6,2:122])

    #  test1=horizontal[5:6,1:119]

    #    test2=horizontal[5:6,1:119]
    #    test3=horizontal[5:6,2:120]
    #    test4=horizontal[5:6,3:121]
    #    test5=horizontal[5:6,4:122]
    # Apply morphology operations
    #  te
    #  st1=cv.erode(test1,horizontalStructure)
    #  print(test1)
    #  show_wait_destroy("test1", test1)

    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)

    # Show extracted horizontal lines
    show_wait_destroy("horizontal", horizontal)

    # [horiz]
    show_wait_destroy("horizontal", horizontal)
    # [vert]
    # Specify size on vertical axis
    print(vertical.shape)
    rows = vertical.shape[0]
    verticalsize = int(rows / 5)

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)

    # Show extracted vertical lines
    show_wait_destroy("vertical", vertical)
    vertical1 = np.copy(vertical)
    # [vert]

    # [smooth]
    # Inverse vertical image
    #    vertical = cv.bitwise_not(vertical)
    #    horizontal = cv.bitwise_not(horizontal)
    # show_wait_destroy("vertical_bit", vertical)
    show_wait_destroy("vertical_bit", vertical)
    '''
    Extract edges and smooth image according to the logic
    1. extract edges
    2. dilate(edges)
    3. src.copyTo(smooth)
    4. blur smooth img
    5. smooth.copyTo(src, edges)
    '''

    # Step 1
    blur1 = cv.GaussianBlur(vertical,(5,5),0);
    _,edges = cv.threshold(blur1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU);
    blur2 = cv.GaussianBlur(horizontal, (5, 5), 0);
    _, edges1 = cv.threshold(blur2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    # edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
    #                              cv.THRESH_BINARY, 3, -2)
    # edges1 = cv.adaptiveThreshold(horizontal, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
    #                               cv.THRESH_BINARY, 3, -2)
    temp = edges & edges1;
    #    show_wait_destroy("edges", edges)
    #
    #    # Step 2
    kernel = np.ones((2, 2), np.uint8)
    temp = cv.erode(temp, kernel)
    # show_wait_destroy("dilate", temp)
    # temp = cv.erode(temp,kernel)
    show_wait_destroy("erode", temp)

    result = np.copy(src)

    result = bw - horizontal
    result = result - vertical
    show_wait_destroy("before final", result)

    result1 = np.copy(bw)
    # show_wait_destroy("before final1", result1)
    # show_wait_destroy("vertical1", vertical1)
    h, w = 520, 900
    result1 = result - vertical1
    show_wait_destroy("vertical1", result1)
    text = list()
    for i in range(horizontal.shape[0]-1):
        for j in range(horizontal.shape[1]):
            #print(horizontal[i][j],horizontal[i+1][j])
            if horizontal[i][j] == 255 and horizontal[i+1][j] == 255:horizontal[i][j] = 0;
    for i in range(vertical.shape[0]):
        for j in range(vertical.shape[1]-1,0,-1):
            if vertical[i][j] == 255 and vertical[i][j-1] == 255:vertical[i][j] = 0;
    i = horizontal & vertical;
    (rows, col) = np.where(i != 0)
    y = np.unique(rows)
    x = np.unique(col)
    print(x)
    print(y)
    config = ("-l eng --oem 1 --psm 6")
    # tessdata_dir_config ='--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
    py.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    datalist = []
    # result = cv.adaptiveThreshold(result, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
    #                                       cv.THRESH_BINARY, 11, 131)
    show_wait_destroy("result",result)
    for i in range(len(y) - 1):
        templist = []
        for j in range(len(x) - 1):
            # if (abs((y[i + 1]) - y[i]) < 5 or abs(x[j + 1] - x[j]) < 5):
            #    continue
            roi = result[y[i]:y[i + 1] + 1, x[j]:x[j + 1] + 1]
            #roi = cv.resize(roi, None, fx=3, fy=3)
            roi = cv.bitwise_not(roi)

           # roigray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            newroi = cv.resize(roi, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)
            # ret, newroi1 = cv.threshold(roigray, 127, 255, cv.THRESH_BINARY_INV)
            # mykernel = np.ones((3, 3), np.uint8)
            # roiClosing = cv.morphologyEx(newroi1, cv.MORPH_CLOSE, mykernel)
            # cv.imwrite("new.jpg", roiClosing)
            # newroi1=cv.threshold(newroi,127,255,cv.THRESH_BINARY)
            #show_wait_destroy("roi", newroi)
            string = (str)(py.image_to_string(newroi,config=config))
            if(ord(string[0]) == 12):string = ""
            templist.append(string)
            #cv.rectangle(src, (x[j], y[i]), (x[j + 1], y[i + 1]), (255, 0, 0), 1)
            #show_wait_destroy("final1", src)
        datalist.append(templist)
    with open('people.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(datalist)
    writeFile.close()

    #    print(rows)
    #
    #    # Step 3
    #    smooth = np.copy(vertical)
    #    show_wait_destroy("test1",smooth)
    #    # Step 4
    #    smooth = cv.blur(smooth, (2, 2))
    #    show_wait_destroy("test1",smooth)
    #    # Step 5
    # (rows, cols) = np.where(edges != 0)
    #    vertical[rows, cols] = smooth[rows, cols]
    #
    #    Show final result
    #    show_wait_destroy("smooth - final", vertical)
    #   [smooth]
    # print(text)

    # show_wait_destroy("final", result)
    show_wait_destroy("final1", result1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 2))
    # kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    fresult = np.copy(result1)
    cv.imwrite("result.jpg", result1)
    cv.imwrite("final.jpg", fresult)
    fresult = cv.erode(fresult, kernel1)

    fresult = cv.dilate(fresult, kernel)
    # fresult=cv.dilate(fresult,kernel)
    show_wait_destroy("final1", fresult)

    contours = cv.findContours(fresult, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]

    show_wait_destroy("final1", src)
    # contours=sorted(contours,key=lambda c:min(max(c[0,:,:])))
    print(len(contours[0]))
    temp = list()

    #    for i in range(len(contours)):
    ##        mask1=np.zeros((h,w),dtype=np.uint8)
    ##        mask=cv.drawContours(mask1,contours,i,255,-1)
    ##        re=cv.bitwise_and(mask1,mask1,mask=mask)
    #        cnt=contours[i]
    #        x,y,w,h=cv.boundingRect(cnt)
    #        temp.append(list((x,y,w,h)))
    #

    temp.reverse()
    for i in range(len(temp)):
        x, y, w, h = tuple(temp[i])
        cv.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return 0

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)



if __name__ == "__main__":
    py.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    img = cv.imread('table2.png',0)
    # thresholding the image to a binary image
    thresh, img_bin = cv.threshold(img, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # inverting the image
    img_bin = 255 - img_bin
    show_wait_destroy("img",img_bin)
    kernel_len = np.array(img).shape[1] // 100
    ver_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    image_1 = cv.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv.dilate(image_1, ver_kernel, iterations=3)
    show_wait_destroy("vertical",vertical_lines)
    image_2 = cv.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv.dilate(image_2, hor_kernel, iterations=3)
    show_wait_destroy("horizontal",horizontal_lines)
    img_vh = cv.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    show_wait_destroy("img_vh",img_vh)
    # Eroding and thesholding the image
    img_vh = cv.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv.threshold(img_vh, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    bitxor = cv.bitwise_xor(img, img_vh)
    bitnot = cv.bitwise_not(bitxor)
    show_wait_destroy("bitnot",bitnot)
    contours, hierarchy = cv.findContours(img_vh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    # Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    # Get mean of heights
    mean = np.mean(heights)
    # Create list box to store all boxes in
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        if (w < 1000 and h < 500):
            image = cv.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            box.append([x, y, w, h])
    show_wait_destroy("image",image)
    # Creating two lists to define row and column in which cell is located
    row = []
    column = []
    j = 0
    # Sorting the boxes to their respective row and column
    for i in range(len(box)):
        if (i == 0):
            column.append(box[i])
            previous = box[i]
        else:
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]
                if (i == len(box) - 1):
                    row.append(column)
            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])
    print(column)
    print(row)
    # calculating maximum number of cells
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol
    center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
    center = np.array(center)
    center.sort()
    finalboxes = []
    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)
    outer = []
    config = ("-l eng --oem 1 --psm 6")
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner =''
            if (len(finalboxes[i][j]) == 0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                 finalboxes[i][j][k][3]
                    finalimg = bitnot[x:x + h, y:y + w]
                    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))
                    border = cv.copyMakeBorder(finalimg, 2, 2, 2, 2, cv.BORDER_CONSTANT, value=[255, 255])
                    resizing = cv.resize(border, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
                    dilation = cv.dilate(resizing, kernel, iterations=1)
                    erosion = cv.erode(dilation, kernel, iterations=1)
                    #show_wait_destroy("erosion",erosion)
                    out = py.image_to_string(erosion,config=config)
                    if (len(out) == 0):
                        out = py.image_to_string(erosion, config=config)
                    if(ord(out[0])==12):out=""
                    inner = inner + " " + out
                outer.append(inner)
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
    dataframe.to_csv("people.csv")
    # data = dataframe.style.set_properties(align="left")
    # data.to_excel("people.xlsx")

