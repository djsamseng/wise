#include <cstddef>
#include <cstdio>
#include <seccomp.h>
#include <unistd.h>

#include <string>
#include <iostream>
#include <vector>

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cublas_utils.h"


// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

typedef float datatype;

__global__
void cuda_add(int n, datatype *x, datatype *y) {
    // which thread inside the grid
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // number of threads in the block * number of blocks in a grid
    int stride = blockDim.x * gridDim.x;
    // Since there are only blockDim.x * gridDim.x threads
    // and only 1 grid we need to use stride to cover the remaining items
    // avoid strides to avoid distance unaligned global memory access (performance hit)
    // shared_memory is shared by all threads in a thread block

    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

__global__
void cuda_add_half(int n, __half *x, __half *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        y[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));
    }
}

__global__
void cuda_add_hadd(int n, __half *x, __half *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        y[i] = __hadd(x[i], y[i]);
    }
}

template<class T>
__global__
void init_vals(int n, T*x, T*y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        x[i] = T(1);
        y[i] = T(2);
    }
}

template <class T>
void run_for_datatype(void kernel(int, T*, T*)) {
    int N = 1 << 20;

    int threadsPerBlock = 256;
    // How many blocks do I need to get N threads?
    // N + threadsPerBlock - 1 to make sure we round up for integer division
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (false) {
        std::cout << "N:" << N
            << " numBlocks:" << numBlocks
            << " threadsPerBlock:" << threadsPerBlock
            << " product:" << numBlocks * threadsPerBlock
            << std::endl;
    }

    T *x, *y;
    cudaMallocManaged(&x, N*sizeof(T));
    cudaMallocManaged(&y, N*sizeof(T));
    init_vals<T><<<numBlocks, threadsPerBlock>>>(N, x, y);

    cudaEvent_t start, mid, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&mid);
    cudaEventCreate(&stop);

    // These three lines occur on the GPU stream
    // without blocking the CPU until
    // cudaEventSynchronize
    cudaEventRecord(start);
    kernel<<<numBlocks, threadsPerBlock>>>(N, x, y);
    cudaEventRecord(mid);
    kernel<<<numBlocks, threadsPerBlock>>>(N, x, y);
    cudaEventRecord(stop);

    cudaError_t cudaerr = cudaEventSynchronize(stop);
    if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    }
    cudaError_t cudaAsyncErr = cudaGetLastError();
    if (cudaAsyncErr != cudaSuccess) {
        printf("Async kernel error: %s\n", cudaGetErrorString(cudaAsyncErr));
    }


    for (int i = 0; i < N; i++) {
        if (unsigned(y[i]) != 4) {
            std::cout
                << "Error at:"
                << i << " value:" << unsigned(y[i])
                << std::endl;
            break;
        }
    }

    float ms1 = 0;
    float ms2 = 0;
    cudaEventElapsedTime(&ms1, start, mid);
    cudaEventElapsedTime(&ms2, mid, stop);

    std::cout << "Finished"
        << " Cuda time first:" << ms1
        << "ms"
        << " Cuda time second:" << ms2
        << "ms"
        << std::endl;
}

void matrix_mult() {
    using data_type = double;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 2;
    const int n = 2;
    const int k = 2;
    const int lda = 2;
    const int ldb = 2;
    const int ldc = 2;
    /*
     *   A = | 1.0 | 2.0 |
     *       | 3.0 | 4.0 |
     *
     *   B = | 5.0 | 6.0 |
     *       | 7.0 | 8.0 |
     */

    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    const std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};
    std::vector<data_type> C(m * n);
    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    printf("A\n");
    print_matrix(m, k, A.data(), lda);
    printf("=====\n");

    printf("B\n");
    print_matrix(k, n, B.data(), ldb);
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: compute */
    CUBLAS_CHECK(
        cublasDgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   C = | 23.0 | 31.0 |
     *       | 34.0 | 46.0 |
     */

    printf("C\n");
    print_matrix(m, n, C.data(), ldc);
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

}

struct QemuSeccompSyscall {
    int32_t num;
    uint8_t priority;
};

// Shift + Alt + select with mouse
static const int32_t seccomp_whitelist[] = {
      SCMP_SYS(poll
    ),SCMP_SYS(futex
    ),SCMP_SYS(ioctl
    ),SCMP_SYS(mmap
    ),SCMP_SYS(munmap
    ),SCMP_SYS(brk
    ),SCMP_SYS(openat
    ),SCMP_SYS(read
    ),SCMP_SYS(close
    ),SCMP_SYS(fcntl
    ),SCMP_SYS(write
    ),SCMP_SYS(socket
    ),SCMP_SYS(fstat
    ),SCMP_SYS(set_robust_list
    ),SCMP_SYS(clone
    ),SCMP_SYS(mprotect
    ),SCMP_SYS(stat
    ),SCMP_SYS(lstat
    ),SCMP_SYS(seccomp
    ),SCMP_SYS(getpid
    ),SCMP_SYS(bind
    ),SCMP_SYS(mkdir
    ),SCMP_SYS(readlink
    ),SCMP_SYS(eventfd2
    ),SCMP_SYS(connect
    ),SCMP_SYS(statfs
    ),SCMP_SYS(getdents64
    ),SCMP_SYS(sysinfo
    ),SCMP_SYS(madvise
    ),SCMP_SYS(listen
    ),SCMP_SYS(setsockopt
    ),SCMP_SYS(uname
    ),SCMP_SYS(sched_getaffinity
    ),SCMP_SYS(get_mempolicy
    ),SCMP_SYS(unlink
    ),SCMP_SYS(geteuid
    ),SCMP_SYS(sched_get_priority_max
    ),SCMP_SYS(sched_get_priority_min
    ),SCMP_SYS(prlimit64
    ),SCMP_SYS(lseek
    ),SCMP_SYS(rt_sigaction
    ),SCMP_SYS(rt_sigprocmask
    ),SCMP_SYS(pread64
    ),SCMP_SYS(access
    ),SCMP_SYS(execve
    ),SCMP_SYS(arch_prctl
    ),SCMP_SYS(set_tid_address
    ),SCMP_SYS(tgkill
    ),SCMP_SYS(pipe2
    )
};

int main() {
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_KILL);

    // To get system calls used
    // strace -qcf ./src/wise-app 2>&1 >/dev/null | awk '{print $NF}'
    for (int i = 0; i < 49; i++) {
        int rc = seccomp_rule_add(ctx, SCMP_ACT_ALLOW, seccomp_whitelist[i], 0);
    }
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(fstat), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(close), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit_group), 0);

    seccomp_load(ctx);
    // for 1 billion (1 << 30)
    // 15ms
    std::cout << "Running for datatype" << std::endl;
    run_for_datatype<datatype>(cuda_add);
    // 7ms
    std::cout << "Running for __half" << std::endl;
    run_for_datatype<__half>(cuda_add_half);
    std::cout << "Running for __hadd" << std::endl;
    run_for_datatype<__half>(cuda_add_hadd);
    std::cout << "Running for __half" << std::endl;
    run_for_datatype<__half>(cuda_add_half);
    std::cout << "Running for __hadd" << std::endl;
    run_for_datatype<__half>(cuda_add_hadd);

    matrix_mult();

    return 0;
}