
import pickle
import numpy as np
from matplotlib import pyplot as plt

dl_arr = np.array(10*np.power(1.65, np.arange(0, 10, 1)), int)
drop_arr = np.arange(0, 1, 0.1)


test = []
train = []
p = []
k = []

for dl in dl_arr:
    test.append([])
    train.append([])
    k.append([])
    p.append([])
    for drop in drop_arr:
        d = pickle.load(open('HyperExpNull/' + 'dl_' + str(dl)[:5] + 'dr_' + str(drop)[:5], 'rb'))
        #d = pickle.load(open('HyperExp/' + 'dl_' + str(dl) + '_dr_' + str(drop), 'rb'))
        s = []
        r = []
        for i in d['stats']:
            #print(i['ml_rmsd'])
            s.append(i['ml_rmsd']['test'])
            r.append(i['ml_rmsd']['train'])
        test[-1].append(np.mean(s))
        train[-1].append(np.mean(r))
        k[-1].append({'dl' : dl, 'drop' : drop})
        p[-1].append(d['params'])

scale = 1.5
        
k = np.array(k)
test = np.array(test)
train = np.array(train)
dif = test-train
d = [0.2, 0.3, 0.4]
print('diff, testval, params')
for i in d:
    m = (test+100*(dif>i)).flatten().argmin()
    v = (test+100*(dif>i)).flatten()[m]
    print(i, v, k.flatten()[m])

f, ax = plt.subplots(3)
plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(test)
ax[0].set_title("test RMSD")
ax[1].imshow(train)
ax[1].set_title("train RMSD")
ax[2].imshow(dif)#np.multiply(dif < 0.4, dif))
ax[2].set_title("overfitting")
plt.gcf().set_size_inches(2*scale, 4*scale)
plt.savefig("hypergraph1.png", dpi=200)
#plt.show()
plt.clf()
print('done')

test, train = [], []
p, k = [], []
cl_arr = np.hstack((np.arange(1, 14), np.arange(16, 40, 3), np.arange(40, 80, 4), np.arange(80, 120, 4)))#cl_arr = np.hstack((np.arange(1, 11, 1), np.arange(1, 65, 5)))
for cl in cl_arr:
    d = pickle.load(open('HyperExpNull2/' + 'cl_' + str(cl)[:5], 'rb'))
    s = []
    r = []
    for i in d['stats']:
        s.append(i['ml_rmsd']['test'])
        r.append(i['ml_rmsd']['train'])
    test.append(np.mean(s))
    train.append(np.mean(r))
    k.append(cl)
    p.append(d['params'])

test, train = np.array(test), np.array(train)

test = test[cl_arr.argsort()]
train = train[cl_arr.argsort()]
cl_arr = cl_arr[cl_arr.argsort()]
print("cl_size, diff, test, train")
for i in range(len(test)):
    print(("   " + str(cl_arr[i]))[-3:], ", ", str(test[i]-train[i])[:5], ", ", str(test[i])[:5], ", ", str(train[i])[:5])

plt.plot(cl_arr, test, c='green', label='test')
plt.plot(cl_arr, train, c='red', label='train')
plt.legend()
plt.xlabel("convolutional layer size")
plt.ylabel("RMSD (kcal/mol)")
plt.gcf().set_size_inches(3*scale, 3*scale)
plt.savefig("hypergraph2.png", dpi=200)
#plt.show()
