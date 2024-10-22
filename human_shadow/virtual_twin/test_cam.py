import pdb
import numpy as np 


fx = 1057.7322998046875
fy = 1057.7322998046875
cx = 972.5150756835938
cy = 552.568359375
v_fov= 54.09259796142578
h_fov= 84.45639038085938
d_fov= 92.32276916503906

img_w = 1920
img_h = 1080


sensor_width_1 = 2 * np.tan(np.deg2rad(h_fov / 2)) * 1
sensor_height_1 = 2 * np.tan(np.deg2rad(v_fov / 2)) * 1

sensor_width_2 = img_w / fx
sensor_height_2 = img_h / fy

sensor_width_3 = 2 * np.tan(np.deg2rad(h_fov / 2)) * fx
sensor_height_3 = 2 * np.tan(np.deg2rad(v_fov / 2)) * fy


print("sensor_width_1", sensor_width_1)
print("sensor_width_2", sensor_width_2)
print("sensor_width_3", sensor_width_3)

print("sensor_height_1", sensor_height_1)
print("sensor_height_2", sensor_height_2)
print("sensor_height_3", sensor_height_3)