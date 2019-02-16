import numpy as np
from scipy.special import expit

with open('/home/zqh/Documents/kendryte-standalone-sdk/src/k210_face/act_ary.h', 'w') as f:
    f.write('#ifndef _ACTARY_H_ \n\
# define _ACTARY_H_ \n\
# include <stdint.h> \n\
float sigmoid[]  = {')
    for i in range(256):
        f.write(str(expit((i*0.07048363031125536)+(-14.773731231689453)))+',')
    f.write('};\n\
# endif')


offset = np.zeros((7, 10, 2))
for i in range(7):
    for j in range(10):
        offset[i, j, :] = np.array([j, i])  # NOTE  [x,y]
offset[..., 0] /= 10
offset[..., 1] /= 7
xy_offset = np.rollaxis(offset, 2, 0)
xy_offset = xy_offset.ravel()

with open('/home/zqh/Documents/kendryte-standalone-sdk/src/k210_face/xy_offset.h', 'w') as f:
    f.write('#ifndef _XYOFFSET_H_ \n\
# define _XYOFFSET_H_ \n\
# include <stdint.h> \n\
float xy_offset[]  = {')
    for off in xy_offset:
        f.write(str(off)+',')
    f.write('};\n\
# endif')
