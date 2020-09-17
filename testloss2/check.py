import numpy as np
predictions=np.empty((3,1,3,3))
a=np.empty(predictions.shape)
a[0]=np.ones((1,1,3,3))
a[1]=np.zeros((1,1,3,3))
a[2]=0.5*np.ones((1,1,3,3))
for i in range(3):
    predictions[i,:,:,:]=a[i]

print(predictions)
import
