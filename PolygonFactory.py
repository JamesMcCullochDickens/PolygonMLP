import math
import numpy as np
from PIL import Image

def getVertices(n_sides, radius):
    vertices = []
    for vertex_num in range(0, n_sides):
        x_val = math.ceil(radius*math.cos((vertex_num*2*math.pi)/n_sides))
        y_val = math.ceil(radius*math.sin((vertex_num*2*math.pi)/n_sides))
        vertices.append((x_val, y_val))
    return vertices

def getPixelsOnLine(vertexA, vertexB):
    dx = vertexB[0] - vertexA[0]
    dy = vertexB[1] - vertexA[1]
    if math.fabs(dx) > math.fabs(dy):
        steps = int(math.fabs(dx))
    else:
        steps = int(math.fabs(dy))
    line_vals = []
    Xincrement = (dx/float(steps))
    Yincrement = (dy/float(steps))
    x = vertexA[0]
    y = vertexA[1]
    line_vals.append((x, y))
    for step in range(steps):
        x = x + Xincrement
        y = y + Yincrement
        x_ = int(round(x))
        y_ = int(round(y))
        line_vals.append((x_, y_))
    return line_vals

def getIm(n_sides, radius, im_width):
    im = np.full((im_width, im_width), 255)
    vertices = getVertices(n_sides, radius)

    # convert the vertices to pixel co-ordinates
    for i in range(len(vertices)):
        vertices[i] = (vertices[i][0] + math.floor(im_width / 2), vertices[i][1] + math.floor(im_width / 2))


    # set the pixel values on the lines to be black
    for index in range(len(vertices)-1):
        pix_vals = getPixelsOnLine(vertices[index], vertices[index+1])
        for pix in pix_vals:
            x_val = pix[0]
            y_val = pix[1]
            im[x_val][y_val] = 0

    # the line from the last vertex to the starting vertex
    pix_vals = getPixelsOnLine(vertices[n_sides-1], vertices[0])
    for pix in pix_vals:
        x_val = pix[0]
        y_val = pix[1]
        im[x_val][y_val] = 0

    for i in range(im_width):
        smallest = 100000000
        largest = -1
        for j in range(im_width):
            if im[i][j] == 0:
                if j < smallest:
                    smallest = j
                if j > largest:
                    largest = j
        if smallest == 100000000:
            continue
        else:
            for t in range(smallest, largest+1):
                im[i][t] = 0

    return im.astype(np.uint8)

def displayPolygon(n_sides, radius, im_width):
    im = getColoredPolygon(n_sides, radius, im_width)
    im = Image.fromarray(im)
    im = im.show()

def getColoredPolygon(n_sides, radius, im_width):
    im = getIm(n_sides, radius, im_width)
    im = np.expand_dims(im, axis=2)
    color_im = np.concatenate((im, im, im), axis=2)
    rand_color = np.random.randint(255, size=3)
    color_im = np.where(color_im == [0, 0, 0], rand_color, color_im)
    return color_im.astype(np.uint8)


def getPolygonArea(n_sides, radius):
    return (1/2)*n_sides*radius**2*math.sin((2*math.pi)/n_sides)

# testing displayPolygon and polygonArea
#triangle = displayPolygon(4, 398, 1200)
#area = getPolygonArea(3, 1)
#print(area)