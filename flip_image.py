import os
import cv2
import re

for root,directories,filenames in os.walk('/home/timonwilli/catkin_ws/src/thymio_course_skeleton/scripts'):
	for filename in [f for f in filenames if f.endswith('.png')]:
    	m = re.search('(\d*)_lin(\d|\d.\d)_ang((-)?(\d|\d.\d))\.png', filename)
    	index = m.group(1)
		angular_vel = m.group(3)

		path = os.path.join(root,filename)
		write_path = os.path.join(root+'_flipped',index+'_lin'+linear_vel+'_ang'+angular_vel+'.png')
        im = cv2.imread(path)
		try:
			im = cv2.flip(im,1)
			cv2.imwrite(write_path,im)
			print('done {0}'.format(path))
		except:
			print('FAIL ON {0}'.format(path))
