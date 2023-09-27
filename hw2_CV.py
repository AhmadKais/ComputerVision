import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter, convolve
from PIL import Image
import os

def transform(image, window_size=3):
    half_window_size = window_size // 2
    image = cv2.copyMakeBorder(image, top=half_window_size, left=half_window_size, right=half_window_size,
                               bottom=half_window_size, borderType=cv2.BORDER_CONSTANT, value=0)

    rows, cols = image.shape
    census = np.zeros((rows - half_window_size * 2, cols - half_window_size * 2), dtype=np.uint8)
    center_pixels = image[half_window_size:rows - half_window_size, half_window_size:cols - half_window_size]

    offsets = [(row, col) for row in range(half_window_size) for col in range(half_window_size) if
               not row == half_window_size + 1 == col]
    for (row, col) in offsets:
        census = (census << 1) | (image[row:row + rows - half_window_size * 2,
                                  col:col + cols - half_window_size * 2] >= center_pixels)
    return census


def column_cost(left_col, right_col):
    return np.sum(np.unpackbits(np.bitwise_xor(left_col, right_col), axis=1), axis=1).reshape(left_col.shape[0],
                                                                                              left_col.shape[1])
def cost(left, right, window_size=3, disparity=0):
    ct_left = transform(left, window_size=window_size)
    ct_right = transform(right, window_size=window_size)
    rows, cols = ct_left.shape
    C = np.full(shape=(rows, cols), fill_value=float('inf'))
    for col in range(disparity, cols):
        C[:, col] = column_cost(ct_left[:, col:col + 1], ct_right[:, col - disparity:col - disparity + 1]).reshape(
            ct_left.shape[0])
    return C

def column_cost_tag(left_col, right_col):
    return np.sum(np.unpackbits(np.bitwise_xor(left_col, right_col), axis=1), axis=1).reshape(left_col.shape[0],
                                                                                              left_col.shape[1])
def cost_tag(left, right, window_size=3, disparity=0):
    ct_left = transform(left, window_size=window_size)
    ct_right = transform(right, window_size=window_size)
    rows, cols = ct_right.shape
    C = np.full(shape=(rows, cols), fill_value=float('inf'))
    for col in range(cols - disparity - 1, 0, -1):
        C[:, col] = column_cost_tag(ct_left[:, col + disparity:col + disparity + 1], ct_right[:, col:col + 1]).reshape(
            ct_right.shape[0])
    return C

def norm(image):
    return cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

def return_depth(left, right, blur_size, window_size, disp, im):
    ct_costs = []
    for disparity in range(0, disp):
        if im == 'left':
            ct_costs.append(cost(left, right, window_size, disparity))
        else:
            ct_costs.append(cost_tag(left, right, window_size, disparity))
    stacked_matrices = np.stack(ct_costs, axis=0)
    depth, h, w = stacked_matrices.shape
    blurred_data = np.zeros_like(stacked_matrices)
    final = np.zeros_like(stacked_matrices)
    for i in range(stacked_matrices.shape[0]):
        blurred_data[i] = gaussian_filter(stacked_matrices[i], sigma=blur_size)
        final[i] = blurred_data[i]
    disparity_map = final.copy()
    disparity_map = np.argmin(disparity_map, axis=0)
    return disparity_map

def ConsistencyTest(disp1, disp2, dir='left'):
    h, w = disp1.shape
    CT = np.zeros((h, w))

    for x in range(w):
        for y in range(h):
            if (dir == 'left'):
                if x - disp1[y, x] >= 0 and x - disp1[y, x] < w:
                    if abs(disp1[y, x] == disp2[y, x - disp1[y, x]]):
                        CT[y, x] = disp1[y, x]
            else:
                if x + disp2[y, x] >= 0 and x + disp2[y, x] < w:
                    if abs(disp2[y, x] == disp1[y, x + disp2[y, x]]):
                        CT[y, x] = disp2[y, x]

    return CT

def save_numpy_array_as_jpg(numpy_array, output_path):
    try:
        # Convert the NumPy array to an image
        image = Image.fromarray(numpy_array)

        # Save the image as a JPEG file
        image.save(output_path, "JPEG")
        print("Image saved successfully as JPEG.")
    except IOError:
        print("Unable to save image.")

def main_func(path):
    current_dir = os.getcwd()
    print(current_dir)
    # Specify the directory name
    new_dir = path

    # Path to the new directory
    new_dir_path = os.path.join(current_dir+'/results', new_dir)
    saving_dir = current_dir+'\\results\\'+path
    print(saving_dir)
    # Create the new directory if it doesn't exist
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    with open(path +'/max_disp.txt', 'r') as file:
        ndisp = int(file.read().strip())
    leftRGB = cv2.imread(path +'/im_left.jpg')
    left = cv2.imread(path +'/im_left.jpg', 0)
    right = cv2.imread(path+'/im_right.jpg', 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)
    disparity_map1 = return_depth(left, right, 11,11, ndisp, 'left')
    disparity_map2 = return_depth(left, right, 11,11, ndisp, 'right')
    disp1 = disparity_map1.astype(np.int16)
    disp2 = disparity_map2.astype(np.int16)
    ConTest = ConsistencyTest(disp1, disp2, 'left')
    ConTest2 = ConsistencyTest(disp1, disp2, 'right')
    disparity_map1 = norm(disp1)
    disparity_map2 = norm(disp2)
    matrix = np.loadtxt('example/K.txt')
    baseline = 0.1  # Example value, adjust according to your setup
    focal_length = matrix[0][0]  # Example value, adjust according to your camera
    left_depth_map = baseline * focal_length / (norm(ConTest)+ 1e-6)
    right_depth_map = baseline * focal_length / (norm(ConTest2)+ 1e-6)
    cv2.imshow('depth_left',((left_depth_map)))
    cv2.imwrite(os.path.join(saving_dir, 'depth_left.png'), 255*left_depth_map)
    cv2.imshow('depth_right',((right_depth_map)))
    cv2.imwrite(os.path.join(saving_dir, 'depth_right.png'), 255*right_depth_map)
    cv2.imshow('disp_left', (norm(ConTest)))
    cv2.imwrite(os.path.join(saving_dir, 'disp_left.png'), norm(ConTest))
    cv2.imshow('disp_right', (norm(ConTest2)))
    cv2.imwrite(os.path.join(saving_dir, 'disp_right.png'), norm(ConTest2))
    # Define camera intrinsic matrix K
    K = np.loadtxt(path+'\K.txt')

    # Initialize empty arrays for 3D points and corresponding 2D points
    points_3d = []
    points_2d = []

    # Perform 2D-3D reprojection
    height, width = left_depth_map.shape
    for v in range(height):
        for u in range(width):
            depth = left_depth_map[v, u]
            if depth > 0:
                point_3d = np.linalg.inv(K) @ np.array([u, v, 1]) * depth
                point_2d = K @ (point_3d / point_3d[2])
                points_3d.append(point_3d)
                points_2d.append(point_2d[:2])

    # Convert the lists of points to NumPy arrays
    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)

    # Perform 3D-2D reprojection
    reprojected_2d = []
    for i in range(points_3d.shape[0]):
        point_2d = K @ (points_3d[i] / points_3d[i, 2])
        reprojected_2d.append(point_2d[:2])
    reprojected_2d = np.array(reprojected_2d)

    # Create a blank image with the same size and channels as the original image
    synthesized_image = np.zeros_like(leftRGB)

    # Copy RGB values from the original image to the synthesized image based on the reprojected 2D coordinates
    for i, point in enumerate(reprojected_2d):
        x, y = point.round().astype(int)
        if 0 <= x < synthesized_image.shape[1] and 0 <= y < synthesized_image.shape[0]:
            synthesized_image[y, x] = leftRGB[y, x]

    K = np.loadtxt(path +'\K.txt')
    # Define camera intrinsic matrix K

    R = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]], dtype=np.float32)

    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    baseline = 1
    points_3d = []
    points_2d = []

    # Perform 2D-3D reprojection
    height, width = left_depth_map.shape
    for v in range(height):
        for u in range(width):
            depth = left_depth_map[v, u]
            if depth > 0:
                point_3d = np.linalg.inv(K) @ np.array([u, v, 1]) * depth
                point_2d = K @ (point_3d / point_3d[2])
                points_3d.append(point_3d)
                points_2d.append(point_2d[:2])

    # Convert the lists of points to NumPy arrays
    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)

    # Create a list of synthesized images
    synthesized_images = []

    # Loop over 11 camera positions
    for i in range(11):
        # Update T by adding the baseline distance to the x-coordinate
        T[0][3] = -0.01 * i
        RT = R @ T
        P = np.matmul(K, RT)
        # Perform 3D-2D reprojection with the updated T
        reprojected_2d = []
        for j in range(points_3d.shape[0]):
            point_3d = np.append(points_3d[j], [1.0])  # Add a fourth coordinate to the 3D point
            # print(point_3d.shape)
            # point_3d = point_3d.reshape((4,1))
            point_3d_world = P @ point_3d  # Multiply T with the 3D point
            # point_3d_world = K @ point_3d_world
            point_2d = point_3d_world / point_3d_world[2]
            reprojected_2d.append(point_2d[:2])
        reprojected_2d = np.array(reprojected_2d)

        # Create a blank image with the same size and channels as the original image
        synthesized_image = np.zeros_like(leftRGB)

        # Copy RGB values from the original image to the synthesized image based on the reprojected 2D coordinates
        for k, point in enumerate(reprojected_2d):
            x, y = point.round().astype(int)
            if 0 <= x < synthesized_image.shape[1] and 0 <= y < synthesized_image.shape[0]:
                synthesized_image[y, x] = leftRGB[y, x]

        # Append the synthesized image to the list
        synthesized_images.append(synthesized_image)

    # Display the synthesized images
    for i, image in enumerate(synthesized_images):
        #cv2.imwrite(f'synthesized_image_{i}.jpg', image)
        cv2.imshow(f"Synthesized Image {i}", image)
        cv2.imwrite(os.path.join(saving_dir, f'synthesized_image_{i}.jpg'), image)

if __name__ == "__main__":
    # Get the current working directory
    current_dir = os.getcwd()

    # Specify the directory name
    new_dir = 'results'

    # Path to the new directory
    new_dir_path = os.path.join(current_dir, new_dir)

    # Create the new directory if it doesn't exist
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    main_func('example')
    main_func('set_1')
    main_func('set_2')
    main_func('set_3')
    main_func('set_4')
    main_func('set_5')
    cv2.waitKey(0)
    cv2.destroyAllWindows()