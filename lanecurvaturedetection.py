
#importing dependencies
import numpy as np
import cv2
import time

#helper function .1
#this function receives each frame(originalimage) as the argument and return a binary image
def thresholdBinary(image, s_thresh=(120, 255), sx_thresh=(20, 120)):
    #hls s-channel thresholding 
    #convert the image into hls color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #keep only the S-channel for thresholding
    S = hls[:,:,2]
    #create a blank image
    s_binary = np.zeros_like(S)
    #keep only the pixel value between min and max s-channel threshold
    s_binary[(S > s_thresh[0]) & (S <= s_thresh[1])] = 1
    
    #Gradient thresholding using sobel filter
    #convert the image into grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #apply sobel filter in x-axis
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    #create a blank image
    sxbinary = np.zeros_like(scaled_sobel)
    #keep only the pixel value between min and max sobel-x threshold
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Combine the two threshold binary image 
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary



#helper funnction no.2
#this function recieves the threshold binary image and convert it into bird's eye view image
def birdseyeView(img):
    img_size = (img.shape[1], img.shape[0])
    #select four source points as four corners of the region of interest
    src = np.float32(
    [[(img_size[0] / 2) - 40, img_size[1] / 2 + 80],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 40), img_size[1] / 2 + 80]])
    #select four destination points to transform region of interest into bird's eye view
    dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
    
    #calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    #Warp with the original image(binary threshold)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)    
    return warped, M
    
#helper funnction no.3
#this function recieves the birdsEyeView image from the first frame as the argument
#and locate the lane lines (coordinate of both left and right lane pixels) 
def locateFirstFrameLane(binary_warped, nwindows = 9, margin = 100, minpix = 50):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)  
    
    midpoint = np.int(histogram.shape[0]/2)
    # Find the peak of the left halves of the histogram
    leftx_base = np.argmax(histogram[:midpoint])
    # Find the peak of the right halves of the histogram
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # These two positions(above) are the base from where we start searching for the lane lines
    # Current positions are two base position which will to be upted later
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Two Python lists(initially empty) for storing left and right lane indecies 
    left_lane_inds = []
    right_lane_inds = []
    
    nonzero = binary_warped.nonzero()
    # y positions of all nonzero pixels in the image
    nonzeroy = np.array(nonzero[0])
    # y positions of all nonzero pixels in the image
    nonzerox = np.array(nonzero[1])
    

    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Set a window height based on the image shape(birdsEyeView)
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Iterate through the n number of windows
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels(x and y)inside the region of the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Add these onformation to the python lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        #if the number of pixels inside the current window is more than 50(minpix)
        #update the base position for the next window(x-axis) on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the list indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions from the list
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Since we are working with curvy line eqn. is x=ay2+by+c 
    # fit a second order polynomial to both left and right line pixel positions
    # which returns the coefficients a,b,c
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
        
    return left_fit, right_fit,left_lane_inds, right_lane_inds, nonzerox, nonzeroy


#helper funnction no.4
#this function recieves the birdsEyeView image(any frame except first frame) as the argument
#and locate the lane lines (coordinate of both left and right lane pixels) 
def locateNextFrameLane(left_fit, right_fit, binary_warped, margin = 50):
    
    nonzero = binary_warped.nonzero()
    # y positions of all nonzero pixels in the current image
    nonzeroy = np.array(nonzero[0])
    # y positions of all nonzero pixels in the current image
    nonzerox = np.array(nonzero[1])
    
    # use the information of left_fit and right_fit(coefficient a,b,c) 
    # and calculate two lists left_lane_inds and right_lane_inds
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
                      & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
                       & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # # Extract left and right line pixel positions from the list
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial left indices (if not empty)
    if len(leftx) == 0:
        left_fit_new =[]
    else:
        left_fit_new = np.polyfit(lefty, leftx, 2)
    # Fit a second order polynomial left indices (if not empty)    
    if len(rightx) == 0:
        right_fit_new =[]
    else:
        right_fit_new = np.polyfit(righty, rightx, 2)
        
    # return the new fitting coefficients         
    return left_fit_new, right_fit_new

#helper function no. 5
# this function receives the polynomial coefficients as arguments and 
# returns the coordinates of detected left and right lane lines
def getCoordinate(warped, left_fit, right_fit):    
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    # Use the 2nd order polynomial coefficients to find the x-coordinates of the lane lines
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return ploty, left_fitx, right_fitx

# helper function no. 6
# this function receives the polynomial coefficients as arguments and 
# returns the radius of the curvature for left and right lane
def radiusOfCurvature(warped, left_fit, right_fit):
    
    ploty, left_fitx, right_fitx = getCoordinate(warped, left_fit, right_fit)
    
    # Convert pixels space into meters
    # meters per pixel in y dimension
    ym = 30/720
    # meters per pixel in x dimension
    xm = 3.7/700 
    y_eval = np.max(ploty)
    
    # Fit new polynomials to new x(ym) and y(xm)
    left_fit_new = np.polyfit(ploty*ym, left_fitx*xm, 2)
    right_fit_new = np.polyfit(ploty*ym, right_fitx*xm, 2)
    
    # Calculate the new radius of curvature
    left_curvature =  ((1 + (2*left_fit_new[0] *y_eval*ym + left_fit_new[1])**2) **1.5) / np.absolute(2*left_fit_new[0])
    right_curvature = ((1 + (2*right_fit_new[0]*y_eval*ym + right_fit_new[1])**2)**1.5) / np.absolute(2*right_fit_new[0])
    return left_curvature, right_curvature

#helper function no. 7
# This function 2nd order polynomial coefficients
# return the decision wheather the lane curve is to the left or right side
def finalDecesion(left_fit, right_fit):
    if(left_fit[0]<0 and left_fit[1]>0):
        return 'Curve to the left'
    elif(right_fit[1]<0 and right_fit[0]>0):
        return 'Curve to the right'

#helper function no. 8
# color the lane area
# write necessary text on the image/fram
def writeOnImage(undist, warped_img, left_fit, right_fit, M, left_curvature, right_curvature, show_values = False):
    
    ploty, left_fitx, right_fitx = getCoordinate(warped, left_fit, right_fit)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw lane on the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,0,255))
    Minv = np.linalg.inv(M)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    #difference = left_curvature-right_curvature
    decesion = finalDecesion(left_fit, right_fit)
    cv2.putText(result, 'Radius of left lane curvature: {:.0f} m'.format(left_curvature), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    cv2.putText(result, 'Radius of left lane curvature: {:.0f} m'.format(right_curvature), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    cv2.putText(result, decesion, (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    
    return result, decesion

# input video to test the result
inputVideo1 = 'assets/inputVideo1.mp4'
inputVideo2 = 'assets/inputVideo2.mp4'
# pipeline for video
# initilize vedio capture
cap = cv2.VideoCapture(inputVideo2)
# initialize a counter 
counter =0
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if (ret==True):
        #skipping frames (working on the even nth frames)
        if(counter%2==0):
            img = frame
            #convert the image into a threshold binary image
            img_binary = thresholdBinary(img)
            #conver the binary image to birdseyeview image
            warped, M = birdseyeView(img_binary)
            #locate the lane lines for first frame
            if(counter==0):
                left_fit, right_fit,left_lane_inds, right_lane_inds, nonzerox, nonzeroy = locateFirstFrameLane(warped)
            #locate the lane lines for after first frame
            else:
                left_fit, right_fit = locateNextFrameLane(left_fit, right_fit, warped)   
             
            #calculate the radius the curvature
            left_curvature, right_curvature = radiusOfCurvature(warped,left_fit,right_fit)      
            result_new, decesion = writeOnImage(img, warped, left_fit, right_fit, M, left_curvature, right_curvature, True)
            #show output(lane and curve detected) on a outputwindow
            print("FPS: ", 1.0 / (time.time() - start_time))
            cv2.imwrite('output2_'+str(counter)+'.jpg',img)
            cv2.imshow('output', result_new) 
            key = cv2.waitKey(1)
            if key == 27:
                break
            counter+=1
        else:
            counter+=1
    else:
        break

cap.release()
cv2.destroyAllWindows()






