import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import requests
import tempfile
import os

class ServiceNode(Node):
    def __init__(self):
        super().__init__('service_node')

        # Initialize service
        self.srv = self.create_service(Trigger, 'run_inference', self.handle_inference)

        # Initialize camera subscriber
        self.bridge = CvBridge()
        self.latest_frame = None
        self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)

        self.get_logger().info(" Service node ready: waiting for /run_inference calls")

    def camera_callback(self, msg):
        """Callback for receiving images from headless_camera."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def handle_inference(self, request, response):
        """Handle Trigger service call to perform inference."""
        self.get_logger().info("Service called: running detection + inference")

        # Ensure a camera frame is available
        if self.latest_frame is None:
            response.success = False
            response.message = "No camera frame available yet"
            return response

        # Save current frame temporarily
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
                temp_path = temp_img.name
                cv2.imwrite(temp_path, self.latest_frame)

            # Send image to inference server
            with open(temp_path, "rb") as f:
                files = {"file": f}
                result = requests.post("http://localhost:8000/infer", files=files)

            # Clean up
            os.remove(temp_path)

            response.success = True
            response.message = f"Inference result: {result.text}"
        except Exception as e:
            response.success = False
            response.message = f"Inference failed: {str(e)}"

        return response


def main(args=None):
    rclpy.init(args=args)
    node = ServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
