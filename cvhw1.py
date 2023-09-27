import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt
import logging


from PIL import Image

MIN_MATCH_COUNT = 10

def puzzle(directory_path,min_match,h,w,transform,ratio):
    image_file = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".jpg")]
    #image_file= glob.glob(directory_path+"/*.jpg")
    #img_file = [image_file.replace("\\", "/") for image_file in image_file]
    img1_path=(directory_path+'piece_1.jpg')
    print(directory_path)
    print((image_file))
    print(img1_path)

    img1 = cv2.imread(img1_path)  # queryImage
    proc_img =[]
    proc_img.append(img1_path )
    #cv2.imshow('nigga3', img1)

    given_affine = transform
    stitched_image = np.zeros((h, w), np.uint8)
    stitched_image = cv2.warpPerspective(img1, given_affine, (h, w))
    img1 = stitched_image
    for img in  image_file :
        if len(image_file)== len(proc_img):
            break
        if img in proc_img:
            continue
        img2 = cv2.imread(img)

        #while(image_file != len(proc_img)):

        #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        #plt.figure()
        #plt.subplot(1, 4, 1)
        #plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
        #plt.title('first image in grey scale')





        #plt.subplot(1, 4, 2)
        #plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
       # plt.title('repositioning img1')



        sift = cv2.SIFT_create()
        #target_kp, target_des = sift.detectAndCompute(img1, None)
        # Compute SIFT keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Initialize and use FLANN
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance <ratio * n.distance:
                good.append(m)
        if len(good) > min_match:
            best_matches = sorted(good, key=lambda x: x.distance)
            pts1 = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
            affine, M = cv2.findHomography(pts2,pts1 ,cv2.RANSAC , 5.0 )
            aligned_img2 = cv2.warpPerspective(img2, affine, (h, w))
            #aligned_img2 = cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2GRAY)
            #plt.subplot(1, 4, 3)

            #plt.imshow(aligned_img2)
            #plt.title(" aligned image after calculating affine trans")
            for m in range (w):
                for n in range (h):
                    if(img1[m,n][0] == 0 & img1[m,n][1] == 0 & img1[m,n][2] == 0):
                        img1[m,n] = aligned_img2[m,n]
            #img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            #plt.subplot(1, 4, 4)
            #plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
            #plt.title("final result")

            #cv2.imshow('nigga ',img1)
            #plt.show()
            proc_img.append(img)

        else:
            print("Not enough matches are found - %d/%d" % (len(good), min_match))
    return img1
if _name_ == '_main_':




    dr="puzzle_homography_1/pieces/"
    tr = np.array([[0.27618429609039, 1.82420249508045, 208.41222081211],
                             [-1.44965374484398, 0.262718789476911, 475.506466804135],
                             [0.000344697162718234, 0.000521151476014606, 0.980040968990257]])

    img = puzzle(dr,10,699,549,tr,0.8)
    cv2.imshow('nigga2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    new_directory = "puzzle_homography_1/result_homography_1"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    image_path = os.path.join(new_directory, "solution_4_4.jpeg")

    # Save the image in JPEG format
    cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])



    dr = "puzzle_homography_2/pieces/"
    tr = np.array([[-0.0708349379518932, 1.47096822969981, 249.354222361031],
                   [-1.58985273334736, 0.277706966241242, 451.148616105541],
                   [-0.00078494806236692, -5.27349121305175e-05, 1.19290397924262]])

    img = puzzle(dr, 10, 722, 513, tr, 0.8)
    cv2.imshow('nigga2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    new_directory = "puzzle_homography_2/result_homography_2"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    image_path = os.path.join(new_directory, "solution_5_5.jpeg")

    # Save the image in JPEG format
    cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


    dr = "puzzle_homography_3/pieces/"
    tr = np.array([[-0.0833117295657903, -1.30144235172996, 428.889552979444],
                   [1.24035456286638, -0.192970892447606, 174.865528657816],
                   [0.000319863012531033, 0.000180584320089234, 0.967873634188587]])

    img = puzzle(dr, 10,760, 502, tr, 0.8)
    cv2.imshow('nigga2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    new_directory = "puzzle_homography_3/result_homography_3"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    image_path = os.path.join(new_directory, "solution_6_6.jpeg")

    # Save the image in JPEG format
    cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
##############################homog3

    dr = "puzzle_homography_4/pieces/"
    tr = np.array([[0.577065911880189 ,	-0.314708970441488 , 303.742849192656],
                           [0.597267733788063,	0.889445122182486	,80.3259317169583],
                           [-0.000193342249994099 ,	0.000834297247185049 ,	0.757850257607461]])

    img = puzzle(dr, 30, 836, 470, tr,0.5)
    cv2.imshow('nigga2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    new_directory = "puzzle_homography_4/result_homography_4"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    image_path = os.path.join(new_directory, "solution_12_12.jpeg")

        # Save the image in JPEG format
    cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ##################homo4
    dr = "puzzle_homography_5/pieces/"
    tr = np.array([[-0.532794149276012, 0.817972810893439, 470.663103411532],
                   [-1.35170437198646, 0.439985425902259, 356.263240516821],
                   [-0.00190401707061561 ,-7.83787376366189e-05, 1.1441360341315]])

    img = puzzle(dr, 30,811, 457, tr, 0.65)
    cv2.imshow('nigga2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    new_directory = "puzzle_homography_5/result_homography_5"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    image_path = os.path.join(new_directory, "solution_16_16.jpeg")

    # Save the image in JPEG format
    cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    dr = "puzzle_homography_6/pieces/"
    tr = np.array([[0.0172285355724898, 0.511676226075184, 157.49915663087],
                   [-0.488577548962103, -0.243107064054147, 153.830317749362],
                   [0.000592988010205602, -0.000364662881777418, 0.599653023103701]])

    img = puzzle(dr, 10, 815, 464, tr, 0.6)
    cv2.imshow('nigga2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    new_directory = "puzzle_homography_6/result_homography_6"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    image_path = os.path.join(new_directory, "solution_21_21.jpeg")

    # Save the image in JPEG format
    cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    dr = "puzzle_homography_7/pieces/"
    tr = np.array([[0.776654939058093 ,	-0.342990127600556,	162.857437017391],
                   [-0.0589237171456944 ,	0.606035810641715,	32.3469304154607],
                   [0.000203821707319 ,	-0.00148140573486976,	0.935502383902999]])

    img = puzzle(dr, 10, 760, 488, tr,0.45)
    cv2.imshow('nigga2', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    new_directory = "puzzle_homography_7/result_homography_7"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    image_path = os.path.join(new_directory, "solution_24_24.jpeg")

    # Save the image in JPEG format
    cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    dr = "puzzle_homography_8/pieces/"
    tr = np.array([[0.697008906808804 ,	-0.707323989777882 ,	164.649295281708],
                   [0.704755833430248 ,	0.602487256155119 ,	184.594221965134],
                 [0.000410380700554334,	-0.000633866589032911,	1.09395230257493]])

    img = puzzle(dr, 20, 760, 499, tr, 0.6)
    cv2.imshow('nigga2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    new_directory = "puzzle_homography_8/result_homography_8"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    image_path = os.path.join(new_directory, "solution_15_34.jpeg")

    # Save the image in JPEG format
    cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    dr = "puzzle_homography_9/pieces/"
    tr = np.array([[21.2140966877792 ,	22.870113686834 ,	2119.96168671555],
                   [2.22898278944479 ,	11.7590184365295 ,	865.184380114046],
                  [0.0219515676713292 ,	0.0271423796274213 ,	3.43242442739833]])

    img = puzzle(dr,10,816, 490, tr, 0.45)
    cv2.imshow('nigga2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    new_directory = "puzzle_homography_9/result_homography_9"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    image_path = os.path.join(new_directory, "solution_1_35.jpeg")

    # Save the image in JPEG format
    cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    dr = "puzzle_homography_10/pieces/"
    tr = np.array([[5.95996619738401, 2.00291151489762	,422.960865997132],
                   [1.55383610848441 ,	2.62018403517017 ,	406.247644077016],
                   [0.00869570700739368 ,	0.001347609062571 ,	1.39495960574422]])

    img = puzzle(dr,40, 759, 506, tr, 0.85)
    cv2.imshow('nigga2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    new_directory = "puzzle_homography_10/result_homography_10"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    image_path = os.path.join(new_directory, "solution_5_59.jpeg")

    # Save the image in JPEG format
    cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])