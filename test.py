import cv2
import numpy as np
import pytest
from main import (
    blur_image,
    canny_dilate_erode,
    convert_to_grayscale,
    draw_rectangle,
    resize_image,
)

OUTPUT_DIR = "output_files/"


def test_blur_image():
    MAIN_IMG = cv2.imread("input_files/cat.jpg")
    # Call the function
    blurred_img = blur_image(MAIN_IMG, (5, 5), 0)

    # Read the saved image
    saved_img = cv2.imread(OUTPUT_DIR + "blurred_image.png")

    # Check if the images are the same
    assert is_the_same(saved_img, blurred_img)


def test_canny_dilate_erode():
    MAIN_IMG = cv2.imread("input_files/cat.jpg")
    # Call the function
    canny_dilated_eroded_img = canny_dilate_erode(MAIN_IMG, 100, 200, (7, 7), 3)
    print(canny_dilated_eroded_img.shape)
    # Read the saved image
    saved_img = cv2.imread(OUTPUT_DIR + "canny_dilated_eroded.png")
    # Make saved img black and white only
    saved_img = cv2.cvtColor(saved_img, cv2.COLOR_BGR2GRAY)

    # Check if the images are the same
    assert is_the_same(saved_img, canny_dilated_eroded_img)


def test_convert_to_grayscale():
    MAIN_IMG = cv2.imread("input_files/cat.jpg")
    # Call the function
    gray_img = convert_to_grayscale(MAIN_IMG)

    # Read the saved image as grayscale
    saved_img = cv2.imread(OUTPUT_DIR + "gray_image.png", cv2.IMREAD_GRAYSCALE)

    # Check if the images are the same
    assert is_the_same(saved_img, gray_img)


def test_draw_rectangle():
    MAIN_IMG = cv2.imread("input_files/cat.jpg")
    # Call the function
    rectangle_img = draw_rectangle(MAIN_IMG, (50, 50), (100, 100), (0, 0, 255), 2)

    # Read the saved image
    saved_img = cv2.imread(OUTPUT_DIR + "rectangle.png")

    # Check if the images are the same
    assert is_the_same(saved_img, rectangle_img)


def test_resize_image():
    MAIN_IMG = cv2.imread("input_files/cat.jpg")
    # Call the function
    resized_img = resize_image(MAIN_IMG, 20)

    # Read the saved image
    saved_img = cv2.imread(OUTPUT_DIR + "resized_image.png")

    # Check if the images are the same
    assert is_the_same(saved_img, resized_img)


def is_the_same(main_img, img_to_compare):
    # Read the images
    img1 = main_img
    img2 = img_to_compare

    # Check if images are of the same size and channels
    if img1.shape != img2.shape:
        print("\n\rThe images have different dimensions.\n\r")
        return False

    # Compute the difference between the images
    difference = cv2.absdiff(img1, img2)

    # If the difference is all zeros, the images are identical
    if not np.any(difference):  # If no difference, images are the same
        print("\n\rThe images are the same.\n")
        return True
    else:
        print("\n\rThe images are different.\n\r")
        return False
