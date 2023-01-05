
def choose_ind(version,sample_size=348,interval=80):
    k = (version-2) % (sample_size//interval + 1)
    if (k+1)*interval > sample_size:
        return sample_size - interval, sample_size
    else:
        return k*interval,(k+1)*interval
