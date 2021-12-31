
import math
import time

import cv2
import mss
import numba.cuda
import numpy as np

def screen_cap(sct):
    # (1080, 3840, 4)
    im = np.asarray(sct.grab(sct.monitors[0]))
    # BGRA
    # np.all(img[:,:,3] == 0) is True, alpha is all 0
    return im

# Takes 11-18ms
# vs c++ takes 0.12-0.24ms
def capture_screen_fast():
    t0 = time.time()
    sct = mss.mss()
    t1 = time.time()
    for _ in range(5):
        t2 = time.time()
        im = screen_cap(sct)
        t3 = time.time()
        print("Mss creation:", t1-t0, "Screen capture:", t3-t2)
    # cv2.imshow("Screen capture:", im)
    # cv2.waitKey(0)


@numba.cuda.jit
def cuda_add_atomic(N, x, y):
    index = numba.cuda.blockIdx.x * numba.cuda.blockDim.x + numba.cuda.threadIdx.x
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for i in range(index, N, stride):
        numba.cuda.atomic.add(y, i, x[i])

@numba.cuda.jit
def cuda_add(N, x, y):
    index = numba.cuda.blockIdx.x * numba.cuda.blockDim.x + numba.cuda.threadIdx.x
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for i in range(index, N, stride):
        y[i] = x[i] + y[i]

def run_for_datatype(datatype, kernel):
    N = 1 << 30
    x = np.ones((N,), dtype=datatype)
    y = np.ones((N,), dtype=datatype) + 1
    print(x.size / 1000 / 1000, " million items")

    threadsPerBlock = 256
    numBlocks = int(math.floor(N + threadsPerBlock - 1) // threadsPerBlock)

    start = numba.cuda.event()
    mid = numba.cuda.event()
    stop = numba.cuda.event()

    t1 = time.time()
    x = numba.cuda.to_device(x)
    y = numba.cuda.to_device(y)
    t2 = time.time()

    start.record()
    kernel[numBlocks,threadsPerBlock](N, x, y)
    mid.record()
    kernel[numBlocks,threadsPerBlock](N, x, y)
    stop.record()

    numba.cuda.synchronize()

    t3 = time.time()
    print(
        "First:", numba.cuda.event_elapsed_time(start, mid), "ms",
        "Second:", numba.cuda.event_elapsed_time(mid, stop), "ms",
        "Transfer time:", (t2-t1)/1000, "ms",
        "Cpu kernel launch time:", (t3-t2)/1000, "ms",
    )

    time.sleep(5)
    x = x.copy_to_host()
    y = y.copy_to_host()
    assert np.all(y == 4) == True

def space_test():
    connectivity = 1_000
    layers = 2_000
    # 1000 layers of 1000 = 4GB VRAM
    # 2000 layers of 1000 = 7.8GB VRAM
    W = np.random.rand(layers, connectivity, connectivity).astype(np.float32)
    W = numba.cuda.to_device(W)
    time.sleep(5)

def main():
    # For float32, 1 billion x's + 1 billion y's takes 15ms and 8GB VRAM
    # vs c++ also takes 15ms
    # however c++ can use float16 which takes 7ms and half the memory
    run_for_datatype(np.float32, cuda_add)
    run_for_datatype(np.float32, cuda_add_atomic)

if __name__ == "__main__":
    #main()
    #space_test()
    capture_screen_fast()

