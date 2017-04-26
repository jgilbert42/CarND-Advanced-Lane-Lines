import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from moviepy.editor import VideoFileClip

output_images_dir = './output_images/'
test_images_dir = './test_images/'
output_video_file = 'output.mp4'

mtx = None
dist = None

def load_image(filename):
    return mpimg.imread(filename)

def calibrate_camera(rows=6, cols=9):
    mtx = None
    dist = None

    save_file = 'calibration.npz'
    try:
        data = np.load(save_file)
        mtx = data['mtx']
        dist = data['dist']
        print('using saved calibration')
    except FileNotFoundError:
        print('begin calibration')
        filenames = glob('camera_cal/*.jpg')

        objpoints = [] # 3D points in real world space
        imgpoints = [] # 2D points in image plane

        #Prepare object points, like (0,0,0), (1,0,0)...
        objp = np.zeros((rows*cols,3), np.float32)
        objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2) # x, y coordinates

        for f in filenames:
            img = load_image(f)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (cols,rows), None)

            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            for f in filenames:
                img = load_image(f)
                undist = cv2.undistort(img, mtx, dist, None, mtx)
                save_output_image(undist, 'undistorted-' + f.split('/')[-1])

            print('end calibration')
            np.savez(save_file, mtx=mtx, dist=dist)

    return mtx, dist

def save_output_image(img, filename, cmap=None):
    mpimg.imsave(output_images_dir + filename, img, cmap=cmap)

def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary

def color_threshold(img):
    #R = img[:,:,0]
    #G = img[:,:,1]
    #B = img[:,:,2]

    #binary = np.zeros_like(R)
    #binary[(R > 200) & (G > 160) & ((B < 100) | (B > 200))] = 1

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]

    binary = np.zeros_like(H)
    binary[(((H > 15) & (H < 24) & (S > 90) & (L > 50)) | (L > 220))] = 1

    return binary

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_lr_window_centroids(image, window_width, window_height, margin):
    #window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    left_centroids = []
    right_centroids = []

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)

    y_base = int(image.shape[0] - window_height/2)

    # Add what we found for the first layer
    y_center = y_base
    left_centroids.append((l_center, y_center))
    right_centroids.append((r_center, y_center))

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        y_center = int(y_base - (level * window_height))

        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_max = np.argmax(conv_signal[l_min_index:l_max_index])
        if l_max > 50:
            left_centroids.append((l_center, y_center))
            l_center = l_max+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_max = np.argmax(conv_signal[r_min_index:r_max_index])
        if r_max > 50:
            right_centroids.append((r_center, y_center))
            r_center = r_max+r_min_index-offset

    return left_centroids, right_centroids

def draw_window_boxes(img, l_points, r_points, window_width, window_height):
    if len(l_points) > 0:
        for p in l_points:
            cv2.rectangle(img, (p[0], p[1]), (p[0] + window_width, p[1] + window_height), (255,0,0), -1)

    if len(r_points) > 0:
        for p in r_points:
            cv2.rectangle(img, (p[0], p[1]), (p[0] + window_width, p[1] + window_height), (0,255,0), -1)

    return img

def draw_window_centroids(warped, window_centroids, window_width = 50, window_height = 80):
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows    
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        #template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(l_points) # create a zero color channle 
        template = np.array(cv2.merge((l_points,r_points,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 0.5, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

    return output

def draw_text(img, text, origin):
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2)

def pipeline_image(img, save_images=None, save_suffix='.jpg'):
    if save_images:
        print('begin pipeline_image', save_suffix)

    undistorted = undistort(img)
    if save_images:
        save_output_image(undistorted, 'undistorted' + save_suffix)

    #binary = abs_sobel_thresh(undistorted, orient='x', sobel_kernel=15, thresh=(20,100))
    binary = color_threshold(undistorted)
    if save_images:
        save_output_image(binary, 'binary' + save_suffix, cmap='gray')

    img_size = binary.shape[::-1]

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

    if save_images:
        cv2.polylines(img, np.int32([src]), True, (255,0,0), thickness=3)
        save_output_image(img, 'polygon' + save_suffix)

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(binary, M, img_size, flags=cv2.INTER_LINEAR)

    if save_images:
        save_output_image(warped, 'warped' + save_suffix, cmap='gray')

    window_width = 40
    window_height = 60

    #identified lane-line pixels and fit their positions with a polynomial
    l_points, r_points = find_lr_window_centroids(warped, window_width, window_height, 100)
    global last_l_points, last_r_points
    if len(l_points) < 5 and len(last_l_points) > 0:
        #print("less than 4 l_points:", len(r_points))
        # use the previous points
        l_points = last_l_points
    else:
        last_l_points = l_points
    l_points = np.array(l_points, dtype=np.int32)
    l_poly = np.polyfit(l_points[:,1], l_points[:,0], 2)

    if len(r_points) < 5 and len(last_r_points) > 0:
        #print("less than 4 r_points:", len(r_points))
        r_points = last_r_points
    else:
        last_r_points = r_points
    r_points = np.array(r_points, dtype=np.int32)
    r_poly = np.polyfit(r_points[:,1], r_points[:,0], 2)

    yval = np.arange(0, warped.shape[0])
    l_xval = np.polyval(l_poly, yval)
    r_xval = np.polyval(r_poly, yval)

    if save_images:
        lanes = warped*255
        lanes = np.array(cv2.merge((lanes,lanes,lanes)),np.uint8) # make window pixels green
        lanes = draw_window_boxes(lanes, l_points, r_points, window_width, window_height)

        for p in l_points:
            cv2.circle(lanes, (p[0], p[1]), 10, (255,0,255), -1)
        for p in r_points:
            cv2.circle(lanes, (p[0], p[1]), 10, (255,0,255), -1)

        for x,y in zip(l_xval, yval):
            cv2.circle(lanes, (int(x),y), 5, (255,255,0), -1)
        for x,y in zip(r_xval, yval):
            cv2.circle(lanes, (int(x),y), 5, (0,255,255), -1)

        save_output_image(lanes, 'lanes' + save_suffix, cmap='gray')

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    #calculated the position of the vehicle with respect to center
    lane_center_offset_m = (warped.shape[1]/2 - (l_xval[-1] + r_xval[-1])/2) * xm_per_pix
    direction = 'Left'
    if lane_center_offset_m > 0:
        direction = 'Right'

    #calculated the radius of curvature of the lane
    y_eval = np.max(yval)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(l_points[:,1]*ym_per_pix, l_points[:,0]*xm_per_pix, 2)
    right_fit_cr = np.polyfit(r_points[:,1]*ym_per_pix, r_points[:,0]*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters

    #Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([l_xval , yval]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([r_xval, yval])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    unwarp = cv2.warpPerspective(color_warp, Minv, img_size, flags=cv2.INTER_LINEAR)
    draw_text(undistorted, "Radius: {:.1f}m {:.1f}m".format(left_curverad, right_curverad), (50, 50))
    draw_text(undistorted, "{:.3f}m {} of Center".format(abs(lane_center_offset_m), direction), (50, 100))
    output = cv2.addWeighted(undistorted, 1, unwarp, 0.4, 0)
    if save_images:
        save_output_image(output, 'output' + save_suffix)

    return output

def process_test_images():
    filenames = glob('test_images/*.jpg')
    #filenames = ['test_images/test2.jpg']
    for f in filenames:
        img = load_image(f)
        img_out = pipeline_image(img, True, '-' + f.split('/')[-1])
        #show_before_after(img, img_out, 'gray')

def process_video(in_file, out_file):
    clip = VideoFileClip(in_file)
    video_clip = clip.fl_image(pipeline_image)
    video_clip.write_videofile(out_file, audio=False)

def show_before_after(before, after, cmap=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax1.imshow(before)
    ax1.set_title('Before')
    ax2.imshow(after, cmap=cmap)
    ax2.set_title('After')
    plt.show()

def show_images(imgs, titles):
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for ax, img, title in zip(axes.flat, imgs, titles):
        ax.imshow(img)
        ax.set_title(title)

    plt.show()


last_l_points = []
last_r_points = []

mtx, dist = calibrate_camera()
process_test_images()
process_video('project_video.mp4', 'output.mp4')
process_video('challenge_video.mp4', 'challenge_output.mp4')
process_video('harder_challenge_video.mp4', 'harder_challenge_output.mp4')

