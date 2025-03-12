
from polymetis import CameraInterface
import cv2
camera_interface = CameraInterface(ip_address="10.100.7.16") # This was the IP of the NUC, please double check.

image = camera_interface.read_once()
cv2.imwrite("image", image)

# robot_interface.go_home()