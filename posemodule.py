import cv2
import mediapipe as mp
import time
import argparse
import math

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        lmList = []  # Initialize lmList here
        if self.results.pose_landmarks and draw:
            for lm_id, lm in enumerate(self.results.pose_landmarks.landmark):
                if lm_id > 10:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([lm_id, cx, cy])
            lmList = sorted(lmList, key=lambda x: x[0])

            connections = self.mpPose.POSE_CONNECTIONS
            for connection in connections:
                startingindex, endingindex = connection
                if startingindex > 10 and endingindex > 10:
                    start_point = tuple(lmList[startingindex-11][1:])
                    end_point = tuple(lmList[endingindex-11][1:])
                    cv2.line(img, start_point, end_point, (255, 255, 255), 2)
            for lm in lmList:
                cv2.circle(img, (lm[1], lm[2]), 3, (0, 0, 255), cv2.FILLED)

        return lmList

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                if id > 10:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
            lmList = sorted(lmList, key=lambda x: x[0])

            if draw:
                # Draw a circle on joint 14 (left wrist) for demonstration purposes
                cv2.circle(img, (lmList[14][1], lmList[14][2]), 3, (255, 0, 0), cv2.FILLED)

        return lmList

    def calculateDistances(self, lmList):
        distances = []
        num_joints = len(lmList)
        for i in range(num_joints):
            for j in range(i+1, num_joints):
                joint_1 = lmList[i]
                joint_2 = lmList[j]
                dist = math.sqrt((joint_2[1] - joint_1[1])**2 + (joint_2[2] - joint_1[2])**2)
                distances.append([joint_1[0], joint_2[0], int(dist)])
        return distances

def process_video(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open {source}")
        return

    pTime = 0
    detector = poseDetector(detectionCon=True, trackCon=True)
    frame_counter = 0  

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.resize(img, (640, 512))
        detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(f"Frame: {frame_counter} - Landmarks: ")

            for lm in lmList:
                print(f"Joint {lm[0]}: ({lm[1]}, {lm[2]})")

            distances = detector.calculateDistances(lmList)
            print(f"Frame: {frame_counter} - Distances:")
            for distance in distances:
                joint1, joint2, dist = distance
                print(f"Distance between Joint {joint1} and Joint {joint2}: {dist:.2f}")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Video", img)

        if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(1) == ord('q'):
            break

        frame_counter += 1  

    cap.release()
    cv2.destroyAllWindows()

def process_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    pTime = 0
    detector = poseDetector(detectionCon=True, trackCon=True)
    frame_counter = 0  

    while True:
        success, img = cap.read()
        if not success:
            break

        detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(f"Frame: {frame_counter} - Landmarks: ")

            for lm in lmList:
                print(f"Joint {lm[0]}: ({lm[1]}, {lm[2]})")

            distances = detector.calculateDistances(lmList)
            print(f"Frame: {frame_counter} - Distances:")
            for distance in distances:
                joint1, joint2, dist = distance
                print(f"Distance between Joint {joint1} and Joint {joint2}: {dist:.2f}")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Webcam", img)

        if cv2.waitKey(1) == ord('q'):
            break

        frame_counter += 1  

    cap.release()
    cv2.destroyAllWindows()

def process_image(source):
    img = cv2.imread(source)
    if img is None:
        print(f"Failed to open {source}")
        return

    detector = poseDetector(detectionCon=True, trackCon=True)
    detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
            print("Landmarks: ")

            for lm in lmList:
                print(f"Joint {lm[0]}: ({lm[1]}, {lm[2]})")

            distances = detector.calculateDistances(lmList)
            print("Distances:")

            for distance in distances:
                joint1, joint2, dist = distance
                print(f"Distance between Joint {joint1} and Joint {joint2}: {dist:.2f}")

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(source, mode):
    if mode == "video":
        process_video(source)
    elif mode == "webcam":
        process_webcam()
    elif mode == "image":
        process_image(source)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Estimation')
    parser.add_argument('--source', type=str, default='./posevideos/image.jpg', help='Path to video file or image file or "webcam" for webcam feed')
    parser.add_argument('--mode', type=str, default='image', help='Mode: "video", "webcam", or "image"')
    args = parser.parse_args()
    main(args.source, args.mode)


