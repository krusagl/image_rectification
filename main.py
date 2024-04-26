import numpy as np
import cv2
import matplotlib.pyplot as plt

def imrect(im1):
# Perform Image rectification on an 3D array im.
# Parameters: im1: numpy.ndarray, an array with H*W*C representing image.(H,W is the image size and C is the channel)
# Returns: out: numpy.ndarray, rectified image。
#   out =im1

    # Make a copy of the input image
    img = im1.copy()
    
    # Convert to grayscale 
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_img = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blurred_img, 50, 100)
    
    # Find outer closed contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = img.copy()
    cv2.drawContours(img_with_contours, contours, -1, (0,255,0), 3)

    # Find the contour with the largest area
    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour
    
    # Draw the largest contour on a blank canvas
    canvas = np.zeros_like(img_gray)
    cv2.drawContours(canvas, [largest_contour], -1, (255, 255, 255), cv2.FILLED)
    
    # Mask the original image with the contour
    # This isolates the object (valid) from the background (black)
    masked_image = cv2.bitwise_and(img, img, mask=canvas)

    # Approximate the largest contour to find the 4 corner points
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Sort the 4 corner points based on their coordinates
    sorted_corners = sorted(approx.reshape(-1, 2), key=lambda x: x[1])
    
    top_corners = sorted(sorted_corners[:2], key=lambda x: x[0])
    bottom_corners = sorted(sorted_corners[2:], key=lambda x: x[0])
    
    # Find the top-left, top-right, bottom-left, and bottom-right corners
    top_left_corner = top_corners[0]
    top_right_corner = top_corners[1]
    bottom_left_corner = bottom_corners[0]
    bottom_right_corner = bottom_corners[1]

    # Define the coordinates of the rectangle corners in the original image
    original_corners = np.float32([top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner])
    img_height, img_width = img.shape[:2]
    
    # Calculate the width and height of the rectangle in the original image
    width = round(np.linalg.norm(top_left_corner - top_right_corner))
    height = round(np.linalg.norm(top_left_corner - bottom_left_corner))
    
    # Fix the top-left corner
    x = original_corners[0,0]
    y = original_corners[0,1]
    
    # Specify the destination corners to warp to
    destination_corners = np.float32([[x, y], [x+width-1, y], [x+width-1, y+height- 1], [x, y+height-1]])
    
    # Compute perspective matrix
    perspective_matrix = cv2.getPerspectiveTransform(original_corners, destination_corners)
    
    # Warp the image
    out = cv2.warpPerspective(img, perspective_matrix, (img_width, img_height))
        
    return (out)
 

if __name__ == "__main__":

    # This is the code for generating the rectified output
    # If you have any question about code, please send email to e0444157@u.nus.edu
    # fell free to modify any part of this code for convenience.
    img_names = ['./data/test1.jpg','./data/test2.jpg','./data/test3.jpg']
    for name in img_names:
        img = cv2.imread(name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        rectificated = imrect(img)
        
        warped_img_rgb = cv2.cvtColor(rectificated, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./data/Result_'+name[7:], warped_img_rgb)