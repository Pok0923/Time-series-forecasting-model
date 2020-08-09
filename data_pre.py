
import numpy as np

feature = 1
n_seq = 6
n_steps = 900
len_test = 24*7
len_time_step = 6
X,Y = [],[]
data_all_mmn = []

def fit(x):
    min = x.min()
    max = x.max()
    print('Min:{}, Max:{}'.format(min, max))
    return min,max

def transform(x,max,min):
    x = 1.0 * (x - min) / (max - min)
    
    return x

def inverse_transform(x,max,min):
    x = x * (max - min) + min
    return x

def get_matrix(index):
    return dataset[get_index[index]]

def check_it(depends):
    for d in depends:
        if d not in get_index.keys():
            return False
    return True

def split_(data):
    dataset = data['data'][:, :, 2]
    min_,max_=fit(data)
    for a in data:
        data_all_mmn.append(transform(a,max_,min_))
    dataset=np.array(data_all_mmn)
    dataset = dataset.reshape((-1,1,100,100))
    dataset = dataset[:,:,40:70,50:80]
    
    index = data['idx'].value.astype(str)
    index = to_datetime(index, format='%Y-%m-%d %H:%M')
    offset_frame = pd.DateOffset(minutes=24 * 60 // 24)
    
    get_index = dict()
    for i, ts in enumerate(index):
        get_index[ts] = i     
    
    depends = [range(1, len_time_step+1)]
    
    i = len_time_step
    while i < len(index):
        Flag = True
        for depend in depends:
            if Flag is False:
                break
            Flag = check_it([index[i] - j * offset_frame for j in depend])

        if Flag is False:
            i += 1
            continue
        x = [get_matrix(index[i] - j * offset_frame) for j in depends[0]]
        y = get_matrix(index[i])
        if len_time_step > 0:
            X.append(np.vstack(x))
        Y.append(y)
        i += 1
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    return X,Y

def train_data(x,y):
    train_x, test_x  = x[:-len_test], x[-len_test:]
    train_y, test_y = y[:-len_test], y[-len_test:]

    train_x = train_x.reshape((train_x.shape[0], n_seq,n_steps, feature))
    test_x = test_x.reshape((test_x.shape[0], n_seq,n_steps, feature))
    train_y = train_y.reshape((train_y.shape[0],n_steps))
    test_y = test_y.reshape((test_y.shape[0],n_steps))
    
    return train_x,test_x,train_y,test_y
