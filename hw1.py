import numpy as np

def open_dat(file):
    x, y = [], []
    with open(file) as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split('\t')
            x.append([1]+k[:-1][0].split())
            y.append(k[-1])

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    return x, y

def sign(x):
    return 1 if x > 0 else -1

def PLA(x, y, w):
    updated = 0
    for xt, yt in zip(x, y):
        yn = np.dot(w, xt.T)
        if sign(yn) != yt:
            w += yt * xt
            updated += 1
    return updated

def Pocket(x, y, w, epoch):
    ys = []
    for _ in range(epoch):
        updated = 0
        for xt, yt in zip(x, y):
            yn = np.dot(w, xt.T)
            if sign(yn) != yt:
                w += yt * xt
                updated += 1
        ys.append((w, updated))
    ys.sort(key=lambda x: x[1])
    return ys[0]

def PocketTest(x, y, w):
    err = 0
    n = len(y)
    for xt, yt in zip(x, y):
        yn = np.dot(w, xt.T)
        if sign(yn) != yt:
            err += 1
    return err / n

if __name__ == '__main__':

    # x, y = open_dat('hw1_15_train.dat')
    # total = 0
    # for s in range(2000):
    #     w = np.zeros(5, dtype=np.float32)
    #     #w = np.random.uniform(0, 1, 5)
    #     total += PLA(x, y, w)
    # print(total)
    # print(total/2000)


    total = 0
    for s in range(2000):
        x, y = open_dat('hw1_18_train.dat')
        w = np.random.uniform(0, 1, 5)
        ys = Pocket(x, y, w, 50)
        x, y = open_dat('hw1_18_test.dat')
        err = PocketTest(x, y, w)
        print(s, err)
        total += err
        # print(ys)
    print(total)
    print(total/2000)