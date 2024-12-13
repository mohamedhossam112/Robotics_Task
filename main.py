import cv2
import numpy as np

# 1 Blur Image using GaussianBlur
def blur_image(img, kernel, sigmaX):
    blurred_img = cv2.GaussianBlur(img, kernel, sigmaX)
    return blurred_img

# 2 Apply Canny, then Dilate, then Erode
def canny_dilate_erode(img, low_threshold, high_threshold, kernel, iterations):
    # Canny 
    canny_img = cv2.Canny(img, low_threshold, high_threshold)
    # Dilate 
    dilated_img = cv2.dilate(canny_img, kernel, iterations=iterations)
    # Erode 
    eroded_img = cv2.erode(dilated_img, kernel, iterations=iterations)
    return eroded_img

# 3. Convert Image color to Gray
def convert_to_grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

# 5. Resize Image
def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    new_dim = (width, height)
    resized_img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
    return resized_img

# 4. Draw Rectangle
def draw_rectangle(img, start_point, end_point, color, thickness):
    # Draw the rectangle on the image
    img_with_rectangle = cv2.rectangle(img, start_point, end_point, color, thickness)
    return img_with_rectangle

# Apply all functions and save output images
if __name__ == "__main__":
    # Load the image
    image = cv2.imread("input_files/cat.jpg")

    # 1 Apply Blur
    blurred_image = blur_image(image, (5, 5), 0)
    cv2.imwrite("output_files/blurred_image.png", blurred_image)
    print("Blurred image saved.")

    # 2 Apply Canny, Dilate, Erode
    canny_dilated_eroded = canny_dilate_erode(image, 100, 200, (7, 7), 3)
    cv2.imwrite("output_files/canny_dilated_eroded.png", canny_dilated_eroded)
    print("Canny, Dilate, and Erode image saved.")

    # 3 Convert to Grayscale
    gray_image = convert_to_grayscale(image)
    cv2.imwrite("output_files/gray_image.png", gray_image)
    print("Grayscale image saved.")
    
    # 5 Resize Image
    resized_image = resize_image(image, 20)
    cv2.imwrite("output_files/resized_image.png", resized_image)
    print("Resized image saved.")

    # 4 Draw Rectangle
    rectangle_image = draw_rectangle(image, (50, 50), (100, 100), (0, 0, 255), 2)
    cv2.imwrite("output_files/rectangle.png", rectangle_image)
    print("Rectangle image saved.")

