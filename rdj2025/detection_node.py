import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import cv2
import numpy as np
import requests
import collections

CAM_TOPIC = '/camera/image_raw'
INFER_URL = "http://localhost:8000/infer"
FRAME_CHECK_COUNT = 3
GREEN_PCT_THRESHOLD = 0.15
MIN_CONTOUR_AREA_RATIO = 0.02
MORPH_KERNEL = (5,5)
SEND_TIMEOUT = 5

class PotatoDetectionNode(Node):
    def __init__(self):
        super().__init__('potato_detection_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, CAM_TOPIC, self.image_callback, 10)
        self.get_logger().info(f"✅ Subscribed to {CAM_TOPIC}")

        # Keep recent frames
        self.frame_history = collections.deque(maxlen=FRAME_CHECK_COUNT)

        # Create unified service
        self.srv = self.create_service(Trigger, 'run_inference', self.handle_run_inference)
        self.get_logger().info("✅ Service '/run_inference' ready (check green + inference)")

    def preprocess(self, frame):
        frame_resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    def compute_green_percentage(self, img_bgr):
        h, w = img_bgr.shape[:2]
        total_area = h * w
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        largest = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest)
        if contour_area < (MIN_CONTOUR_AREA_RATIO * total_area):
            return 0.0

        mask_inside = np.zeros_like(mask)
        cv2.drawContours(mask_inside, [largest], -1, 255, thickness=-1)
        green_inside = cv2.bitwise_and(mask, mask_inside)
        green_pixels = cv2.countNonZero(green_inside)
        green_pct = green_pixels / (contour_area + 1e-6)
        return float(green_pct)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.frame_history.append(frame)
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")

    def call_inference(self, img_bgr):
        ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not ok:
            self.get_logger().error("❌ JPEG encode failed")
            return None

        files = {'file': ('frame.jpg', jpg.tobytes(), 'image/jpeg')}
        try:
            res = requests.post(INFER_URL, files=files, timeout=SEND_TIMEOUT)
            if res.status_code == 200:
                return res.json()
            else:
                self.get_logger().warn(f"⚠️ Inference returned {res.status_code}")
                return None
        except Exception as e:
            self.get_logger().error(f"❌ Inference call error: {e}")
            return None

    def handle_run_inference(self, request, response):
        if len(self.frame_history) < FRAME_CHECK_COUNT:
            response.success = False
            response.message = "❌ Not enough frames captured yet."
            return response

        greens = []
        for frame in list(self.frame_history):
            pre = self.preprocess(frame)
            greens.append(self.compute_green_percentage(pre))

        avg_green = sum(greens) / len(greens)
        self.get_logger().info(f"🟢 Green% per frame: {[round(g,3) for g in greens]} | Avg = {avg_green:.3f}")

        if all(g >= GREEN_PCT_THRESHOLD for g in greens):
            self.get_logger().info("🌿 Plant detected — running inference...")
            result = self.call_inference(self.frame_history[-1])
            if result and "prediction" in result:
                response.success = True
                response.message = f"🧠 Prediction: {result['prediction']} (green avg {avg_green:.3f})"
            else:
                response.success = False
                response.message = "❌ Inference failed or invalid response."
        else:
            response.success = True
            response.message = f"🚫 No plant detected (avg green% = {avg_green:.3f}) → no_plant"

        return response


def main(args=None):
    rclpy.init(args=args)
    node = PotatoDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
