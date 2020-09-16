import os



def get_gpustat():
    stat = os.popen('gpustat').readlines()[1:]
    for id, gpu in enumerate(stat):
        list1 = []
        index = 0
        index_rate = []
        for i, c in enumerate(gpu):
            if c == '|':
                list1.append(i)
            if c == '/':
                index = i
            if c == "%":
                index_rate = i
        mem_rate = float(gpu[list1[1]+1 : index])/float(gpu[index+1 : list1[2]-3])
        rate = float(gpu[index_rate-5: index_rate])/100
        if mem_rate<0.3 and rate < 0.3:
            return id
    return -1

if __name__ == "__main__":
    print(get_gpustat())









