import matplotlib.pyplot as plt
import align_pdb_to_box as align
import numpy as np 

i = 0
sum_data = np.zeros((16,16,16))
for x in align.generate_dataset("train"):
    data, *torsions = x
    i+=1
    sum_data += data
    if i > 50: 
        break
    
average_data = sum_data / i
    
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
index_list = np.arange(0,16)
X,Y,Z = np.meshgrid(index_list, index_list, index_list)

ax.scatter(X, Y, Z, s=100*average_data, c=average_data)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.savefig("box.png")