import tensorflow as tf
import numpy as np


import os, cv2, re



def get_data(root_dir, stacked=1,grey=True):
    dataset = []
    labels = []
    buffer = [0]
    labels_buf = [0]
    for root, dirs, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('png'):
                try:
                    if grey:
                        im = cv2.imread(os.path.join(root,filename),0)
                        im = im.reshape(im.shape[0],im.shape[1],1)
                        channels = 1
                    else:
                        im = cv2.imread(os.path.join(root,filename))
                        channels = im.shape[2]
                    m = re.search('(\d*)_lin(\d|\d.\d)_ang((-)?(\d|\d.\d))\.png', filename)
                    index = int(m.group(1))
                    linear_vel = float(m.group(2))
                    angular_vel = float(m.group(3))
                    buffer.append(im)
                    labels_buf.append(angular_vel)
                    if len(buffer) == stacked+1:
                        del buffer[0]
                        del labels_buf[0]
                        c = channels*stacked
                        dx = np.zeros((im.shape[0],im.shape[1],c))
                        dy = 0
                        for i in range(stacked):
                            c = i*channels
                            dx[:,:,c:c+channels] = np.array(buffer[i])
                            dy += labels_buf[i]/stacked
                        dataset.append(dx)
                        labels.append(dy)
                except:
                    print('problem with {0}'.format(os.path.join(root,filename)))
                    print('skipping..')

    return np.array(dataset), np.array(labels)
