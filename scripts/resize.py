import os
import cv2

for root,directories,filenames in os.walk('/home/timonwilli/catkin_ws/src/thymio_course_skeleton/scripts'):
    for filename in [f for f in filenames if f.endswith('.png')]:
        path = os.path.join(root,filename)
        im = cv2.imread(path)
	try:
		im = cv2.resize(im,(84,84))
		cv2.imwrite(path,im)
		print('done {0}'.format(path))
	except:
		print('FAIL ON {0}'.format(path))


        
