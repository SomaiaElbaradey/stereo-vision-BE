import cv2

def feature_detection_and_matching(left_img, right_img):
    if len(left_img.shape) == 3:
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_left = left_img

    if len(right_img.shape) == 3:
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_right = right_img

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray_left, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_right, None)

    if descriptors1 is None or descriptors2 is None or len(keypoints1) < 2 or len(keypoints2) < 2:
        return None, []

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    num_matches_to_draw = min(50, len(good_matches))

    matched_image = cv2.drawMatches(
        left_img, keypoints1,
        right_img, keypoints2,
        good_matches[:num_matches_to_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return matched_image, good_matches, keypoints1, keypoints2