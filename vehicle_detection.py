import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from tracking import Tracking
import random

def display_2_images(img1, img2, text_1='Origin Image', text_2='Destination Image'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(text_1, fontsize=50)
    ax2.imshow(img2)
    ax2.set_title(text_2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.interactive(False)


def display_color_gray(color, gray):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(color)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(gray, cmap='gray')
    ax2.set_title('Dest Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.interactive(False)

# always read image as uint8 so don't have to worry png(0-1) or jpeg
def read_image(image_path):
    img = mpimg.imread(image_path)
    if img.dtype != 'uint8':
        img = (img * 255).astype(np.uint8)
    return img

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
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
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def convert_color(image, color_space='RGB'):
    # Assume input is RGB
    feature_image = np.copy(image)

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

    return feature_image


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
        # Read in each one by one convert 0-1 to 0-255 as well
        image = read_image(file)
        # apply color conversion if other than 'RGB'
        feature_image = convert_color(image, color_space)

        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


def normalize_data(car_features, noncar_features):
    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    return X_scaler.transform(X), X_scaler


def train_classifier(car_features, noncar_features, visualize=False):
    # define labels
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

    # Normalize data
    scaled_X, X_scaler = normalize_data(car_features, noncar_features)

    # good idea to split data into training and validation and random
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    # Time to train it
    svc = LinearSVC()
    svc.fit(X_train, y_train)

    if visualize:
        prediction = svc.score(X_test, y_test)
        print('Test Accuracy of SVC = ', prediction)
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        return svc, X_scaler, prediction
    else:
        return svc, X_scaler

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
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

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(img, color_space)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    img_tosearch = img[ystart:ystop, :, :]
    # assume img_tosearch is in RGB format
    ctrans_tosearch = convert_color(img_tosearch, color_space=color_space)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)


    bbox_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bbox_list.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                # cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                #               (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    # print(bbox_list)

    return bbox_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def experiment_color(car_images, noncar_images, colors, spatials, histbins):
    combine = []
    for c in colors:
        for s in spatials:
            for h in histbins:
                d = {}
                d['color_space'] = c
                d['spatial'] = s
                d['histbin'] = h
                combine.append(d)

    print(len(combine))

    # color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    # spatial = 16
    # histbin = 32
    # orient = 9
    # pix_per_cell = 4
    # cell_per_block = 2
    # hog_channel = 1

    best = 0
    best_combination = []
    for c in combine:
        car_features = extract_features(car_images[:999], color_space=c['color_space'],
                                        spatial_size=(c['spatial'], c['spatial']), hist_bins=c['histbin'],
                                        # orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                        # hog_channel=hog_channel,
                                        spatial_feat=True, hist_feat=True, hog_feat=False)
        noncar_features = extract_features(noncar_images[:999], color_space=c['color_space'],
                                           spatial_size=(c['spatial'], c['spatial']), hist_bins=c['histbin'],
                                           # orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                           # hog_channel=hog_channel,
                                           spatial_feat=True, hist_feat=True, hog_feat=False)

        print('Predict for', c)
        classifier, pred = train_classifier(car_features, noncar_features, visualize=True)
        if pred > best:
            best_combination = []
            best = pred
            best_combination.append(c)
        elif pred == best:
            best_combination.append(c)

    print('best combination is',best, best_combination)

def experiment_hog(car_images, noncar_images, colors, orients, pix_per_cells, cell_per_blocks, hog_channels):
    combine = []
    for color in colors:
        for o in orients:
            for p in pix_per_cells:
                for c in cell_per_blocks:
                    for h in hog_channels:
                        d = {}
                        d['color_space'] = color
                        d['orient'] = o
                        d['pix_per_cell'] = p
                        d['cell_per_block'] = c
                        d['hog_channel'] = h
                        combine.append(d)

    print(len(combine))

    # color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    # orient = 9
    # pix_per_cell = 4
    # cell_per_block = 2
    # hog_channel = 1

    best = 0
    best_combination = []
    for c in combine:
        car_features = extract_features(car_images[:999], color_space=c['color_space'],
                                        # spatial_size=(c['spatial'], c['spatial']), hist_bins=c['histbin'],
                                        orient=c['orient'], pix_per_cell=c['pix_per_cell'], cell_per_block=c['cell_per_block'],
                                        hog_channel=c['hog_channel'],
                                        spatial_feat=False, hist_feat=False, hog_feat=True)
        noncar_features = extract_features(noncar_images[:999], color_space=c['color_space'],
                                           # spatial_size=(c['spatial'], c['spatial']), hist_bins=c['histbin'],
                                           orient=c['orient'], pix_per_cell=c['pix_per_cell'], cell_per_block=c['cell_per_block'],
                                           hog_channel=c['hog_channel'],
                                           spatial_feat=False, hist_feat=False, hog_feat=True)

        print('Predict for', c)
        classifier, pred = train_classifier(car_features, noncar_features, visualize=True)
        if pred > best:
            best_combination = []
            best = pred
            best_combination.append(c)
        elif pred == best:
            best_combination.append(c)

    print('best combination is',best, best_combination)

    # Predict for {'histbin': 32, 'color_space': 'HLS', 'spatial': 16}
    # Test Accuracy of SVC =  0.9975


def save_state(svc, X_scaler, pred, color_space, hist_bins, spatial, orient,
    pix_per_cell, cell_per_block, hog_channel, save_file):
    """
    Utility method to save state with pickle
    """

    data = {
        'svc': svc,
        'X_scaler': X_scaler,
        'pred': pred,
        'color_space': color_space,
        'hist_bins': hist_bins,
        'spatial': spatial,
        'orient': orient,
        'pix_per_cell': pix_per_cell,
        'cell_per_block': cell_per_block,
        'hog_channel': hog_channel
    }

    with open(save_file, 'wb') as f:
        pickle.dump(data, f)
        f.close()

def train_classifier_pipeline(save_file='training.p'):
    start = timer()
    car_images = glob.glob('./data/car/*/*.png')
    noncar_images = glob.glob('./data/noncar/*/*.png')

    # train classifier
    color_space = 'YCrCb'
    hist_bins = 16
    spatial = (16,16)
    orient = 9
    pix_per_cell = 4
    cell_per_block = 2
    hog_channel = 'ALL'
    car_features = extract_features(car_images, color_space=color_space, spatial_size=spatial, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    noncar_features = extract_features(noncar_images, color_space=color_space, spatial_size=spatial, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

    svc, X_scaler, pred = train_classifier(car_features, noncar_features, visualize=True)

    #save it so we can use it later
    save_state(svc, X_scaler, pred, color_space, hist_bins, spatial, orient, pix_per_cell, cell_per_block, hog_channel, save_file)

    end = timer()
    print('Duration', round(end - start, 2), 'secs')


def detection_pipeline(image_detection, params, svc, X_scaler, color_space,
    orient, pix_per_cell, cell_per_block, spatial, hist_bins, visualize=False):

    # Method 1 : search everything this is very slow and inefficient
    # images = glob.glob('./test_images/*.jpg')
    # for image in images:
    #     test_image = read_image(image)
    #     window_list_s = slide_window(test_image,y_start_stop=[400,450], xy_window=(32,32), xy_overlap=(0.8,0.8))
    #     window_list_m = slide_window(test_image,y_start_stop=[400,600], xy_window=(128,128), xy_overlap=(0.8,0.8))
    #     window_list_l = slide_window(test_image,y_start_stop=[450,None], xy_window=(256,256), xy_overlap=(0.8,0.8))
    #     windows = window_list_s + window_list_m + window_list_l
    #
    #     hot_windows = search_windows(test_image, windows, svc, X_scaler, color_space=color_space,
    #                                  spatial_size=spatial, hist_bins=hist_bins,
    #                                  orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
    #                                  hog_channel=hog_channel)
    #
    #     window_image = draw_boxes(test_image, hot_windows, color=(0,0,255))
    #     plt.imshow(window_image)
    #     plt.show()

    # Method 2: Get list of boxes being identified by HOG and sub-sampling
    bbox_list = []
    for param in params:
        ystart = param[0]
        ystop = param[1]
        scale = param[2]
        bbox = find_cars(image_detection, ystart, ystop, scale, svc, X_scaler,
                         color_space, orient, pix_per_cell, cell_per_block,
                         spatial, hist_bins)
        bbox_list += bbox

    # search and detect with heapmap
    heatmap_threshold = 2  # how many boxes enough to keep it
    heatmap = np.zeros_like(image_detection[:, :, 0]).astype((np.float))
    heatmap = add_heat(heatmap, bbox_list)
    heatmap = apply_threshold(heatmap, heatmap_threshold)
    heatmap = np.clip(heatmap, 0, 255)
    labels = label(heatmap)

    draw_img = draw_labeled_bboxes(np.copy(image_detection), labels)

    if visualize:
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()

    return draw_img


def process_pipeline(color_image):
    # (400, 528, 0.8), (400, 592, 1.5), (450, 656, 1.7) (Large one hasn't tested)
    scale_list = [(400, 528, 0.8), (400, 592, 1.5), (450, 656, 1.7)]
    image_with_box = detection_pipeline(color_image, scale_list, svc, X_scaler,
                                        color_space, orient, pix_per_cell,
                                        cell_per_block, spatial, hist_bins,
                                        visualize=False)
    return image_with_box


def process_video_pipeline(image_detection):
    scale_list = [(400, 528, 0.8), (400, 592, 1.5), (450, 656, 1.7)]

    bbox_list = []
    for param in scale_list:
        ystart = param[0]
        ystop = param[1]
        scale = param[2]
        bbox = find_cars(image_detection, ystart, ystop, scale, svc, X_scaler,
                         color_space, orient, pix_per_cell, cell_per_block,
                         spatial, hist_bins)
        bbox_list += bbox

    # search and detect with heapmap
    heatmap_threshold = 2  # how many boxes enough to keep it
    heatmap = np.zeros_like(image_detection[:, :, 0]).astype((np.float))
    heatmap = add_heat(heatmap, bbox_list)
    heatmap = apply_threshold(heatmap, heatmap_threshold)
    heatmap = np.clip(heatmap, 0, 255)
    tracking.heatmap_list.append(heatmap)
    tracking.heatmap_list = tracking.heatmap_list[-tracking.number_frames_kept:]
    # combine all heap_list
    combined_heatmap = np.zeros_like(image_detection[:, :, 0]).astype(
        (np.float))
    for h in tracking.heatmap_list:
        combined_heatmap = combined_heatmap + h

    combined_heatmap = apply_threshold(combined_heatmap,
                                       heatmap_threshold * tracking.number_frames_kept // 2)

    labels = label(combined_heatmap)
    tracking.number_of_dections = labels[1]
    # print("This frame detect", tracking.number_of_dections,
    #       len(tracking.heatmap_list))
    draw_img = draw_labeled_bboxes(np.copy(image_detection), labels)
    return draw_img

def save_image(img):
    rand = str(random.random())[-5:]
    mpimg.imsave('./saved_video_images/' + rand + '.jpg', img)
    return img

def generate_video():
    input_video = 'project_video.mp4'
    output_video = 'vehicle_detection.mp4'
    clip = VideoFileClip(input_video)
    processed_clip = clip.fl_image(process_video_pipeline)

    # NOTE: this function expects color images!!
    processed_clip.write_videofile(output_video, audio=False)

# Main function
if __name__ == '__main__':
    from timeit import default_timer as timer

    # Good idea to train once and save for later use
    # train_classifier_pipeline('training.p')

    # Load saved training
    saved_pickle = pickle.load(open('training_ALL_YCrCb.p', 'rb'))
    svc = saved_pickle['svc']
    X_scaler = saved_pickle['X_scaler']
    color_space = saved_pickle['color_space']
    spatial = saved_pickle['spatial']
    hist_bins = saved_pickle['hist_bins']
    orient = saved_pickle['orient']
    pix_per_cell = saved_pickle['pix_per_cell']
    cell_per_block = saved_pickle['cell_per_block']
    hog_channel = saved_pickle['hog_channel']

    # Test images
    # images = glob.glob('./test_images/*.jpg')
    # for image in images:
    #     final_img = process_pipeline(image)
    #     plt.imshow(final_img)
    #     plt.show()

    tracking = Tracking()
    generate_video()
