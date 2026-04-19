from Zandpack.plot import PI, DM, J, plt
import numpy as np
idx1 = np.arange(8)*4
idx2 = np.arange(8)*4 + 1
idx3 = np.arange(8)*4 + 2
idx4 = np.arange(8)*4 + 3

t,  pi = PI(['SCBenzene_2_save'])
tt, dm = DM(['SCBenzene_2_save'])
#ttt,j  = J(['Benzene2_save'])
print(pi.shape)
plt.plot(t[1:], np.trace(pi[:,0,0,idx1,:][:, :, idx1],axis1=1,axis2=2), color='r')
#plt.plot(t[1:], np.trace(pi[:,0,1,idx1,:][:, :, idx1],axis1=1,axis2=2))
plt.plot(t[1:], np.trace(pi[:,0,0,idx2,:][:, :, idx2],axis1=1,axis2=2),color='g')
#plt.plot(t[1:], np.trace(pi[:,0,1,idx2,:][:, :, idx2],axis1=1,axis2=2))
plt.plot(t[1:], np.trace(pi[:,0,0,idx3,:][:, :, idx3],axis1=1,axis2=2),color='y')
#plt.plot(t[1:], np.trace(pi[:,0,1,idx3,:][:, :, idx3],axis1=1,axis2=2))
plt.plot(t[1:], np.trace(pi[:,0,0,idx4,:][:, :, idx4],axis1=1,axis2=2),color='b')
#plt.plot(t[1:], np.trace(pi[:,0,1,idx4,:][:, :, idx4],axis1=1,axis2=2))

#plt.plot(ttt,j[0], linestyle='dashed')
#plt.plot(ttt,j[1], linestyle='dashed')

plt.show()

