# Sky segmentation from image/video
Traditional image processing based sky segmentation.

## Dependencies
- Python3
- OpenCV – 3.4.15
- numpy - 1.12.0

## Algotithm
1. Convert image to greyscale
2. Change the contrast, alpha, gamma of the image to make dark pixels darker, light pixels lighter
3. Use Otsu's threshold method to find the centrer intensity pixel value adaptively
4. Do a region growing contour detection from the top-center pixel value across the width and find all contours with value >=threshold
5. Return the largest contour(sky region)
