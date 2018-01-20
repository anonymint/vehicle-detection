import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

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
        else:
            feature_image = np.copy(image)

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
    X_scaller = StandardScaler().fit(X)
    return X_scaller.transform(X)


def train_classifier(car_features, noncar_features, visualize=False):
    # define labels
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

    # Normalize data
    scaled_X = normalize_data(car_features, noncar_features)

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
        return svc, prediction
    else:
        return svc

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


# Main method for running

def experiment_color(colors, spatials, histbins):
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

def experiment_hog(colors, orients, pix_per_cells, cell_per_blocks, hog_channels):
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

if __name__ == '__main__':
    from timeit import default_timer as timer

    start = timer()
    car_images = glob.glob('./data/car/*/*.png')
    noncar_images = glob.glob('./data/noncar/*/*.png')

    # experiment_color(['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'], [8,16,32], [8,16,32])
    # best combination is 1.0 [{'color_space': 'RGB', 'histbin': 32, 'spatial': 8}, {'color_space': 'HSV', 'histbin': 16, 'spatial': 8}, {'color_space': 'HSV', 'histbin': 16, 'spatial': 16}, {'color_space': 'LUV', 'histbin': 32, 'spatial': 8}, {'color_space': 'HLS', 'histbin': 32, 'spatial': 8}, {'color_space': 'HLS', 'histbin': 32, 'spatial': 16}, {'color_space': 'YUV', 'histbin': 16, 'spatial': 8}, {'color_space': 'YUV', 'histbin': 32, 'spatial': 16}, {'color_space': 'YCrCb', 'histbin': 16, 'spatial': 8}, {'color_space': 'YCrCb', 'histbin': 32, 'spatial': 8}, {'color_space': 'YCrCb', 'histbin': 32, 'spatial': 16}]
    experiment_hog(['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'], [9,12], [4], [2], [0,1,2,'ALL'])
    # best combination is 1.0 [{'orient': 12, 'pix_per_cell': 4, 'color_space': 'RGB', 'hog_channel': 'ALL', 'cell_per_block': 2}, {'orient': 9, 'pix_per_cell': 4, 'color_space': 'HSV', 'hog_channel': 'ALL', 'cell_per_block': 2}, {'orient': 12, 'pix_per_cell': 4, 'color_space': 'HSV', 'hog_channel': 'ALL', 'cell_per_block': 2}, {'orient': 12, 'pix_per_cell': 4, 'color_space': 'LUV', 'hog_channel': 'ALL', 'cell_per_block': 2}, {'orient': 9, 'pix_per_cell': 4, 'color_space': 'HLS', 'hog_channel': 'ALL', 'cell_per_block': 2}, {'orient': 12, 'pix_per_cell': 4, 'color_space': 'HLS', 'hog_channel': 'ALL', 'cell_per_block': 2}, {'orient': 12, 'pix_per_cell': 4, 'color_space': 'YUV', 'hog_channel': 'ALL', 'cell_per_block': 2}, {'orient': 12, 'pix_per_cell': 4, 'color_space': 'YCrCb', 'hog_channel': 'ALL', 'cell_per_block': 2}]

    end = timer()
    print('Duration', round(end - start, 2), 'secs')
