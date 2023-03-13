import cv2
import numpy as np

upper_left = 0
upper_right = 0
lower_left = 0
lower_right = 0


# ex 11

def final_vizualization(left_top, left_bottom, right_top, right_bottom, frame_points, polygon_points,original_frame):
    new_frame_left = np.zeros((240, 426), dtype=np.uint8)
    cv2.line(new_frame_left, left_top, left_bottom, (255, 0, 0), 3)

    matrix_for_unstretching = cv2.getPerspectiveTransform(frame_points, polygon_points)
    unstreched_matrix_left = cv2.warpPerspective(new_frame_left, matrix_for_unstretching, (width, height))

    half1_coordinates = np.argwhere(unstreched_matrix_left > 254)
    left_xs = []
    left_ys = []
    for coordinates in half1_coordinates:
        left_xs.append(coordinates[1])
        left_ys.append(coordinates[0])

    cv2.imshow('new_frame_left', unstreched_matrix_left)

    new_frame_right = np.zeros((240, 426), dtype=np.uint8)
    cv2.line(new_frame_right, right_top, right_bottom, (255, 0, 0), 3)

    unstreched_matrix_right = cv2.warpPerspective(new_frame_right, matrix_for_unstretching, (width, height))
    half2_coordinates = np.argwhere(unstreched_matrix_right > 254)

    right_xs = []
    right_ys = []
    for coordinates in half2_coordinates:
        right_xs.append(coordinates[1])
        right_ys.append(coordinates[0])

    cv2.imshow('new_frame_right', unstreched_matrix_right)

    for index in range(0,len(left_xs)):
        cv2.circle(original_frame, (left_xs[index],left_ys[index]) ,radius=1, color= (50, 50, 250), thickness=-1)


    for index in range(0,len(right_xs)):
        cv2.circle(original_frame, (right_xs[index],right_ys[index]) ,radius=1, color= (50, 250, 50), thickness=-1)

    cv2.imshow('final_frame', original_frame)


# ex 10

def detect_edges_of_lane(binary_view, left_xs, left_ys, right_xs, right_ys, left_top_x1, left_bottom_x1, right_top_x1,
                         right_bottom_x1):
    # a
    left_line = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
    right_line = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    # b
    left_top_y = height
    if -(10 ** 8) <= left_top_x1 <= 10 ** 8:
        left_top_x1 = (height - left_line[0]) / left_line[1]

    left_bottom_y = 0
    if -(10 ** 8) <= left_bottom_x1 <= 10 ** 8:
        left_bottom_x1 = (0 - left_line[0]) / left_line[1]

    right_top_y = height
    if -(10 ** 8) <= right_top_x1 <= 10 ** 8:
        right_top_x1 = (height - right_line[0]) / right_line[1]

    right_bottom_y = 0
    if -(10 ** 8) <= right_bottom_x1 <= 10 ** 8:
        right_bottom_x1 = (0 - right_line[0]) / right_line[1]

    # c
    left_top = int(left_top_x1), int(left_top_y)
    left_bottom = int(left_bottom_x1), int(left_bottom_y)
    right_top = int(right_top_x1), int(right_top_y)
    right_bottom = int(right_bottom_x1), int(right_bottom_y)

    # d
    cv2.line(binary_view, left_top, left_bottom, (200, 0, 0), 5)
    cv2.line(binary_view, right_top, right_bottom, (100, 0, 0), 5)
    cv2.imshow('Detect-Edges', binary_view)

    return left_top, left_bottom, right_top, right_bottom, left_top_x1, left_bottom_x1, right_top_x1, right_bottom_x1


# ex 9

def get_coordinates(binary_view, width, height):
    # a
    binary_view_copy = binary_view.copy()
    five_percent_left = int(width * 0.05)
    five_percent_right = int(width - width * 0.95)

    binary_view_copy[0:height, 0:five_percent_left] = 0
    binary_view_copy[0:height, width - five_percent_right:width] = 0

    cv2.imshow('No-Noise', binary_view_copy)

    # b
    half1 = binary_view_copy[:, :width // 2]
    half2 = binary_view_copy[:, width // 2:]

    cv2.imshow('Half1', cv2.convertScaleAbs(half1))
    cv2.imshow('Half2', cv2.convertScaleAbs(half2))

    half1_coordinates = np.argwhere(half1 > 254)
    half2_coordinates = np.argwhere(half2 > 254)

    # half2_coordinates[0:height, 1:2] = half2_coordinates[0:height, 1:2] + width // 2
    for coordinate in half2_coordinates:
        coordinate += width // 2 - 35
        # coordinate+=width//2

    left_xs = []
    left_ys = []
    for coordinates in half1_coordinates:
        left_xs.append(coordinates[1])
        left_ys.append(coordinates[0])

    right_xs = []
    right_ys = []
    for coordinates in half2_coordinates:
        right_xs.append(coordinates[1])
        right_ys.append(coordinates[0])

    return left_xs, left_ys, right_xs, right_ys


# ex 8

def binarize_frame(edge_detection_view):
    edge_detection_view[edge_detection_view < 127] = 0
    edge_detection_view[edge_detection_view >= 127] = 255
    binary_view = edge_detection_view
    cv2.imshow('Binary-Matrix', binary_view)
    return binary_view


# ex 7

def edge_detection(blur_view):
    sobel_vertical = np.float32([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])
    sobel_horizontal = np.transpose(sobel_vertical)
    blur_view_float32 = np.float32(blur_view)
    blur_view_float32_1 = blur_view_float32
    blur_view_float32_2 = blur_view_float32

    edge_detection_view_1 = cv2.filter2D(blur_view_float32_1, -1, sobel_vertical)
    edge_detection_view_2 = cv2.filter2D(blur_view_float32_2, -1, sobel_horizontal)

    edge_detection_view_final = np.sqrt(edge_detection_view_1 ** 2 + edge_detection_view_2 ** 2)
    cv2.imshow('Edge-Detection', cv2.convertScaleAbs(edge_detection_view_final))

    return cv2.convertScaleAbs(edge_detection_view_final)


# ex 6

def blur_image(top_down_view):
    blur_view = cv2.blur(top_down_view, ksize=(7, 7))
    cv2.imshow('Blur', blur_view)
    return blur_view


def get_polygon_points():
    upper_left = (int(width) * 0.43, int(height * 0.78))
    upper_right = (int(width) * 0.57, int(height * 0.78))
    lower_right = (width, height)
    lower_left = (0, height)

    return upper_left, upper_right, lower_left, lower_right


# ex 5

def stretch(street_trapezoid, width, height):
    upper_right_frame = (width, 0)
    lower_right_frame = (width, height)
    lower_left_frame = (0, height)
    upper_left_frame = (0, 0)

    upper_left, upper_right, lower_left, lower_right = get_polygon_points()
    polygon_points = np.array([upper_right, lower_right, lower_left, upper_left], dtype=np.int32)
    frame_points = np.array([upper_right_frame, lower_right_frame, lower_left_frame, upper_left_frame], dtype=np.int32)

    polygon_points = np.float32(polygon_points)
    frame_points = np.float32(frame_points)

    matrix_for_stretching = cv2.getPerspectiveTransform(polygon_points, frame_points)

    street_trapezoid = cv2.warpPerspective(street_trapezoid, matrix_for_stretching, (width, height))
    cv2.imshow('Top-Down', street_trapezoid)
    return street_trapezoid, frame_points, polygon_points


# ex 4

def draw_trapezoid_and_streetonly():
    img = np.zeros((240, 426), dtype=np.uint8)
    upper_left = (int(width) * 0.43, int(height * 0.78))
    upper_right = (int(width) * 0.57, int(height * 0.78))
    lower_right = (width, height)
    lower_left = (0, height)
    polygon_points = np.array([upper_right, lower_right, lower_left, upper_left], dtype=np.int32)
    cv2.fillConvexPoly(img, polygon_points, 1)
    cv2.imshow('trapezoid', img * 255)
    cv2.imshow('road', img * frame)
    return img * frame


cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')
left_top_x = 0
left_bottom_x = 0
right_top_x = 0
right_bottom_x = 0
while True:
    height = 240
    width = 426
    ret, frame = cam.read()

    # ex2, resize-ul
    frame = cv2.resize(frame, (426, 240))
    original_frame = frame.copy()

    # ex3, alb-negru
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret is False:
        break

    cv2.imshow('Original', frame)

    # ex4 trapez si strada
    street_trapezoid = draw_trapezoid_and_streetonly()

    # ex5
    top_down_view, frame_points, polygon_points = stretch(street_trapezoid, width, height)

    # ex6
    blur_view = blur_image(top_down_view)

    # ex7
    edge_detection_view = edge_detection(blur_view)

    # ex8
    binary_view = binarize_frame(edge_detection_view)

    # ex9
    left_xs, left_ys, right_xs, right_ys = get_coordinates(binary_view, width, height)

    # ex10
    left_top, left_bottom, right_top, right_bottom, left_top_x, left_bottom_x, right_top_x, right_bottom_x = detect_edges_of_lane(
        binary_view, left_xs, left_ys, right_xs,
        right_ys, left_top_x, left_bottom_x, right_top_x,
        right_bottom_x)
    # ex11
    final_vizualization(left_top, left_bottom, right_top, right_bottom, frame_points, polygon_points, original_frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cam.release()

cv2.destroyAllWindows()
