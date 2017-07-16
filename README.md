# Advanced Lane Finding Using Sliding Window Search
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undist_img.jpg "Undistorted"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./output_images/warp_processing_check.jpg "Warp Example"
[image5]: ./output_images/polynom_windows.jpg "Window Search Example"
[image6]: ./output_images/assembled_img.jpg "Assembled Image"
[image7]: ./output_images/video_thumbnail.jpg "Video Thumbnail"

## Camera Calibration

All cameras are prone to distortion, with the most obvious (deliberate) effect being fish-eyes. To get an accurate read on the images in our lane video, distortion correction must be applied.
Udacity has kindly provided checkerboard images for distortion correct.

I begin by creating a camera class, that stores information about the camera, in particular its matrix and distortion coefficients.
These are necessary for undistorting an image warped by a camera's lens.
Using OpenCV's `cv2.findChessboardCorners()` function, I'm able to prepare two lists containing `objpoints` (points in 3d space) and `imgpoints` (points in 2d space.)
I then use `cv2.calibrateCamera` to return the camera's matrix and distortion coefficients. OpenCV also provides `cv2.undistort` to undistort our test image.
```python
    def calibrate_camera(self, imgList):
        counter = 0
        for img in imgList:
            # Prepare object points (0,0,0), (1,0,0), etc.
            objp = np.zeros((nx*ny,3), np.float32)
            objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

            # Converting to grayscale
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Finding chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            if ret == True:
                self.imgpoints.append(corners)
                self.objpoints.append(objp)
                counter+=1
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        return self.mtx, self.dist
```
```python
    def undistort(self, img):
        return cv2.undistort(img,self.mtx,self.dist,None,self.mtx)
```
(Python notebook cell 4)

Here's an example of the results before and after.

![alt text][image1]

## Pipeline (single images)

### Colour and Gradient Thresholding
I used a combination of color and gradient thresholds to generate a binary image of the lanes. (ipynb cell 6)
You'll notice in the python notebook multiple functions implemented for this purpose, including sobel, magnitude, directional, S-channel and B-channel thresholding.
I ultimately only used S-channel thresholding, and B-channel for better yellow lane detection. This combination offered a good result as shown below.
More could be done for dynamic thresholding based on road surfacing, but this approach was sufficient for this project.
  

### Perspective Transform
Contained in the function pers_transform() (ipynb cell 6). This function grabs 4 points on the lane image that encompass
the lane, and warps it onto a flat plane, transforming our perspective of the road to a bird's eye view.
```python
def pers_transform(img, nx=9, ny=6):
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[190, 720], [582, 457], [701, 457], [1145, 720]])
    offset = [150,0]
    dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset, 
                      np.array([src[3, 0], 0]) - offset, src[3] - offset])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    # Return the resulting image and matrix
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, M, Minv
```

The following is an example of some test images warped and filtered to show only their lane lines. Note how the lane curvature
corresponds to its original image.

![alt text][image4]

### Window Searching and Margin Searching
When finding lane lines for the first time or whenever the lane-finding pipeline is uncertain about lane line locations, it performs a window search. (ipynb cell 10)

A window search takes a histogram of the lower half of the binary image, revealing the neighbourhood about where the lane lines begin.
I then split the image into 9 horizontal slices, sliding a search window across each slice, finding areas of highest frequency.
Once this is found for both left and right lanes, I perform a 2nd-order `np.polyfit()` to get a best fit curve on the line.
I then store this information in the `Line()` class for use later. (ipynb cell 12)

Window searching can be computationally expensive. To save time for subsequent frames, we can perform a margin search. Based on the same principles,
it only searches in a tight margin around the previously established lane line. (ipynb cell 11)

![alt text][image5]

### Radius of Curvature
This is calculated in the `update_lane()` step (ipynb cell 13). Radius of curvature is implemented using this tutorial (http://www.intmath.com/applications-differentiation/8-radius-curvature.php).


### Drawing the Lane
Finally drawing the lane onto the original image is done in `draw_line()` (ipynb cell 15). Here is an example of all images assembled together.

![alt text][image6]

---

### Final Video

[![Advanced Lane Lines][image7]](https://www.youtube.com/watch?v=e3tx_GHIX6M&feature=youtu.be "Advanced Lane Lines")

---

### Discussion

My solution largely followed the methods laid out in the lesson, with some changes.
As suggested by a fellow student, I used the S and B channels to filter my lane lines, which produced a much cleaner result.

I also added the `Line()` and `Camera()` classes to better manage the lane and camera information.

Much like project 1, my implementation could have benefited from first analysing the overall lane type in each frame, and then dynamically switching thresholds/filters to best cater to each surface type.
This pipeline is also rather slow, which would work poorly in a real-time situation. Perhaps lower resolution video or pared down thresholding methods could improve runtime.