import cv2
import numpy as np

class CoinRecognizer:
    def __init__(self):
        self.sift = cv2.SIFT_create(2000)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=500))

        # Load coin template images and compute their keypoints and descriptors
        self.coin_templates = {
            "penny": self._load_and_compute("penny.jpeg"),
            "quarter": self._load_and_compute("quarter.jpg"),
            "dime": self._load_and_compute("dime.jpg"),
            "nickel": self._load_and_compute("nickel.jpg")
        }

    def _load_and_compute(self, filename):
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return keypoints, descriptors

    def detect_circles(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 3)
        circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, 1, blurred_image.shape[0] / 8,
                                   param1=100, param2=30, minRadius=1, maxRadius=100)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(image, center, 1, (0, 100, 100), 3)
                cv2.circle(image, center, radius, (255, 0, 255), 3)
        return image, circles

    def recognize_coins(self, image, circles):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        total_value = 0

        for circle in circles[0, :]:
            x, y, r = circle
            roi = gray_image[y - r: y + r, x - r: x + r]
            kp_img, des_img = self.sift.detectAndCompute(roi, None)

            match_counts = {}
            for coin, (kp_coin, des_coin) in self.coin_templates.items():
                matches = self.flann.knnMatch(des_img, des_coin, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
                match_counts[coin] = len(good_matches)

            if not any(match_counts.values()):
                continue

            best_match = max(match_counts, key=match_counts.get)
            if best_match == "penny":
                total_value += 0.01
            elif best_match == "quarter":
                total_value += 0.25
            elif best_match == "dime" and match_counts["dime"] >= 10:
                total_value += 0.1
            elif best_match == "nickel":
                total_value += 0.05

        return round(total_value, 2)

def main():
    recognizer = CoinRecognizer()
    web_cam = cv2.VideoCapture(0)

    while True:
        ret, frame = web_cam.read()

        if not ret:
            break

        cv2.imshow('VisionCoin', frame)
        key = cv2.waitKey(1)

        if key == ord("c"):
            while True:
                ret, frame = web_cam.read()
                img_with_circles, detected_circles = recognizer.detect_circles(frame)
                cv2.imshow('Circles Plot', img_with_circles)
                key = cv2.waitKey(1)
                if key == 27:
                    break

            total_value = recognizer.recognize_coins(img_with_circles, detected_circles)
            print(total_value)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_with_text = cv2.putText(img_with_circles, "Total Value - " + f'{total_value}$', (10, 25), font, 1, (255, 255, 255), 2)
            cv2.imshow('VisionCoin', img_with_text)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif key == 27:
            break

    web_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()