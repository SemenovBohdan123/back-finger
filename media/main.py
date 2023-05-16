import cv2
import os

def checkUser(source_image_path):
    source_image = cv2.imread(source_image_path)
    score = 0
    file_name = None
    image = None
    kp1, kp2, mp = None, None, None

    for file in os.listdir("./media/images/"):
        target_image = cv2.imread("./media/images/" + file)

        sift = cv2.SIFT.create()
        kp1, des1 = sift.detectAndCompute(source_image, None)
        kp2, des2 = sift.detectAndCompute(target_image, None)
        matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10),dict()).knnMatch(des1, des2, k=2)
        # точки які знаходяться на однаковій відстані
        mp = []
        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                mp.append(p)
        keypoints = 0
        if len(kp1) <= len(kp2):
            keypoints = len(kp1)
        else:
            keypoints = len(kp2)

        if len(mp) / keypoints * 100 > score:
            score = len(mp) / keypoints * 100
            file_name = file
            image = cv2.drawMatches(source_image, kp1, target_image, kp2, mp, None)
            image = cv2.resize(image, None, fx=2.5, fy=2.5)
            cv2.imwrite( './media/check_img.BMP', image)

    return score, file_name

