
from polymetis import CameraInterface
import cv2
# This was the IP of the NUC, please double check.
# 
ip_address = "101.6.103.171"
camera_interface = CameraInterface(ip_address=ip_address) #"10.100.7.16"

image, timestamp = camera_interface.read_once()
cv2.imwrite("image", image)

# robot_interface.go_home()