import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()
#cap = cv2.VideoCapture(0)

#while(True):
image = cv2.imread("ThanhABC_1.jpg")
#    ret, image = cap.read()

result = detector.detect_faces(image)

img=image

for person in result:
    bounding_box = person['box']
    keypoints = person['keypoints']
    
 #   cv2.rectangle(image,
 #                 (bounding_box[0], bounding_box[1]),
  #                (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
  #                (0,155,255),
  #                2)
 #   cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
 #   cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
 #   cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
 #   cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
 #   cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

    img = image[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]
cv2.imshow("image",img)
cv2.imwrite("Thanh_5.jpg",img)

    

    
# When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()
