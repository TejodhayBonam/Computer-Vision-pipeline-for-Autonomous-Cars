import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import copy
import math
from scipy import ndimage as ndi
from skimage.feature import hog, blob_doh, peak_local_max
from skimage.morphology import watershed, disk
from skimage.filters import rank, gaussian_filter
from skimage.util import img_as_ubyte
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    on_windows = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)

    return on_windows


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    img_features = []
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins, 
                                    bins_range=hist_range)
        img_features.append(hist_features)
    if hog_feat == True:
    # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        img_features.append(hog_features)

    # Return list of feature vectors
    return np.concatenate(img_features)

# Convert windows to heatmap numpy array.
def create_heatmap(windows, image_shape):
    background = np.zeros(image_shape[:2])
    for window in windows:
        background[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
    return background

# find the nonzero areas from a heatmap and
# turn them to windows
def find_windows_from_heatmap(image):
    hot_windows = []
    # Threshold the heatmap
    thres = 15
    image[image <= thres] = 0
    # Set labels
    labels = ndi.label(image)
    # iterate through labels and find windows
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        hot_windows.append(bbox)
    return hot_windows

def combine_boxes(windows, image_shape):
    hot_windows = []
    image = None
    if len(windows)>0:
        # Create heatmap with windows
        image = create_heatmap(windows, imagfe_shape)
        # find boxes from heatmap
        hot_windows = find_windows_from_heatmap(image)
    # return new windows
    return hot_windows

# Define a class to receive the characteristics of each line detection
class Window():
    def __init__(self):
        self.probability = []

# Calculate the distance between two points
# didn't do sqrt to shorten the time
def calc_distance(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def find_center(box):
    startx, starty = box[0]
    endx, endy = box[1]
    # Find the center of the box
    return ((startx+endx)/2., (starty+endy)/2.)

# Find the width and the height of the box
def find_radius(box):
    startx, starty = box[0]
    endx, endy = box[1]
    return (endx - startx)/2, (endy - starty)/2

# Create an array for the center and the radius of the boxes
def initialize_center_box(boxes):
    result = []
    for box in boxes:
        center = find_center(box)
        width, height = find_radius(box)
        move = (0, 0) # movement of an object
        result.append((center, width, height, move, 1))
    return result

def sanity_check(old_center, new_center, old_width, new_width,
    old_height, new_height):
    # check if the centers of the boxes are close and 
    # widths and heights are similar.
    if calc_distance(old_center, new_center) < 5000 and \
                abs(old_width - new_width) < 50 and \
                abs(old_height - new_height) < 50:
        return True
    else:
        return False

# smoothing the change of center locations
def average_centers(new_center, old_center):
    w = 2. # weight parameter
    return ((new_center[0]+old_center[0]*w)/(w+1),
            (new_center[1]+old_center[1]*w)/(w+1))

# calculate the movement of an object
def calculate_move(new_center, old_center, old_move):
    w = 6. # weight parameter
    return ((new_center[0]-old_center[0]+w*old_move[0])/(w+1),
        (new_center[1]-old_center[1]+w*old_move[1])/(w+1))

# move the center by move value
def add_center_move(center, move):
    return (center[0]+move[0]/5, center[1]+move[1]/5)

# Add an array for the center and the radius of the boxes
def add_center_box(new_boxes, old_boxes):
    fresh_boxes = [] # initiate the fresh boxes
    max_confidence = 40 # the prob won't exceed this value
    temp_new_boxes = copy.copy(new_boxes)
    w = 3 # weight parameter
    # iterate through old_boxes to see if a similar window is
    # found in new boxes. If found, increase the confidence level by 1
    # If not, then decrease it by 1
    for old_box in old_boxes:
        old_center, old_width, old_height, old_move, old_prob = old_box
        new_boxes = copy.copy(temp_new_boxes)
        if old_prob > 10:
            add_prob = 2
        else:
            add_prob = 1
        found = False
        for new_box in new_boxes:
            new_center, new_width, new_height, new_move, new_prob = new_box
            if sanity_check(old_center, new_center, old_width, new_width, old_height, new_height):
                fresh_box = [average_centers(new_center, old_center), 
                            (new_width+w*old_width)/(w+1), 
                            (new_height+w*old_height)/(w+1), 
                            calculate_move(new_center, old_center, old_move),
                            min(max_confidence,old_prob+add_prob)]
                # remove the new box from an array
                temp_new_boxes.remove(new_box)
                found = True
                break
        # if no new_box is found, subtract the confidence by 1
        if not found:
            fresh_box = [add_center_move(old_center, old_move), old_width, old_height, old_move, old_prob - 1]
        # add the fresh box
        fresh_boxes.append(fresh_box)
    # append the leftover new boxes to old boxes
    fresh_boxes += temp_new_boxes
    # delete if prob = 0
    temp_fresh_boxes = copy.copy(fresh_boxes)
    for box in fresh_boxes:
        if box[-1] <= 0:
            temp_fresh_boxes.remove(box)
    # return the updated old_boxes
    return temp_fresh_boxes

# Compare the new boxes with boxes from previous frames.
def average_boxes(hot_windows, old_boxes, 
                  image_shape):
    hot_boxes = initialize_center_box(hot_windows)
    new_boxes = add_center_box(hot_boxes, old_boxes)
    # print('new_boxes', new_boxes)
    filtered_boxes = []
    for new_box in new_boxes:
        # Draw boxes only if the confidence level is above 1
        if new_box[-1] > 2:
            filtered_boxes.append(new_box)
    new_windows = []
    # convert center-width-height to lefttop-rightbottom format
    for filtered_box in filtered_boxes:
        new_center, new_width, new_height,new_move, new_prob = filtered_box
        new_windows.append(((int(new_center[0]-new_width), int(new_center[1]-new_height)), 
            (int(new_center[0]+new_width), int(new_center[1]+new_height))))
    # Create a heatmap
    heatmap = create_heatmap(new_windows, image_shape)
    # Check if there is any overlap of windows
    if np.unique(heatmap)[-1] >= 2:
        labels = ndi.label(heatmap)[0]
        heatmap_2 = np.zeros_like(heatmap)
        heatmap_2[heatmap>=2] = 1
        labels_2 = ndi.label(heatmap_2)
        array_2 = np.argwhere(labels_2[0])
        for car_number in range(1, labels_2[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels_2[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            num = labels[nonzero[0][0], nonzero[1][0]]
            labels[labels == num] = 0
        heatmap = labels + heatmap_2
        new_windows = find_windows_from_heatmap(heatmap)

    # return the boxes with high confidence and new set of probability array
    return new_windows, new_boxes
