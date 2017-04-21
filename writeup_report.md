**Advanced Lane Finding Project**

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

[image1]: ./output_images/undistorted-calibration4.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image2b]: ./output_images/undistorted-test1.jpg
[image3]: ./output_images/binary-test1.jpg "Binary Example"
[image4]: ./output_images/warped-straight_lines1.jpg "Warp Example"
[image5]: ./output_images/lanes-test6.jpg "Fit Visual"
[image6]: ./output_images/output-test4.jpg "Output"
[video1]: ./output.mp4 "Output Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!  All of the code is in the lane-finder.py file.

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in the `calibrate_camera` method on line 18.  The
calibration mtx and dist are saved to a file for repeat usage.

I start by preparing "object points", which will be the (x, y, z) coordinates
of the chessboard corners in the world. Here I am assuming the chessboard is
fixed on the (x, y) plane at z=0, such that the object points are the same for
each calibration image.  Thus, `objp` is just a replicated array of
coordinates, and `objpoints` will be appended with a copy of it every time I
successfully detect all chessboard corners in a test image.  `imgpoints` will
be appended with the (x, y) pixel position of each of the corners in the image
plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera
calibration and distortion coefficients using the `cv2.calibrateCamera()`
function.  I applied this distortion correction to the test image using the
`cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I have provided an original and distortion corrected
image.  The code is one line 201 in the first part of the `pipeline_image`
method.

![original][image2]
![distortion corrected][image2b]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a color threshold on an HSL converted image primarily focused on hues in
the yellow range and bright colors (white), line 79 `color_threshold` method.
Here's an example of my output for this step.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is on lines 227 through 229.
I chose to hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
While the lines are somewhat parallel, there's some room for improvement.  I do
not expect this is currently the most significant area to improve.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the convolution approach to identifying the lane lines.  This is
implemented in the find_lr_window_centroids method on line 102.  This method
also filters out points with a low number of pixels in the window.  On line
241, these points are used to fit second order polynomials for the left and
right lane lines.  These polynomials are then used to generate lane curves on
line 244.

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Using the example code from the lectures, I did this in lines 268 through 284
in my code in `lane-finder.py`.  These values are drawn onto the output image
at line 300.  The calculated values seem potentially more accurate on the solid
yellow line as opposed to the dashed line.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 288 through 302 in my code in
`lane-finder.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's my [video1]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

While the current implementation works ok on the project video, it doesn't do
very well on the challenge videos.  Improving the binary step to include some
gradients would possibly help.  There currently isn't any smoothing or
averaging of past lines.  This may help to reduce the peaking out at the road
reflectors.  Another possibility for avoiding the road reflectors may be to
adjust the windowing in the convolution step.

