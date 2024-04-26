**Image Rectification**

Image rectification is the process of rectifying an image distorted by affine transform due to viewpoint, which is often seen in document scanning and digital photography of painting. 
This project implements an image rectification algorithm to warp an input image where an object with rectangular boundary is prominently present with a monochorome background.
The resulting rectified images will see the boundaries of the object of interest are either along horizontal or along vertical direction of the image.

**Summary of Process**

After reading the input image in grayscale format, Gaussian blur is applied to reduce noise.
Canny edge detection is performed and outer closed contours are identified using `cv2.findContours`.
As the prominent object, we identify it by finding the contour with the largest area.
A mask of the original image is applied on the largest contour, which isolates the object from the background.
An approximation is made to find the four corner points of the largest contour.
The width and height of the object is then approximated from the corner points.
By fixing one point, we find the warp destination points by adjusting the remaining three points based on the calculated width and height.
The image is then warped using the `cv2.warpPerspective`.
