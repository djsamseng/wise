
#include <dlfcn.h>
#include <iostream>
#include <mylib.h>

#include <chrono>
#include <thread>

const char MYLIB_PATH[] = "/home/samuel/dev/wise/load_library/lib/build/libmylib.so";

int open_large_dl() {
    // 0.19ms to open a 2GB .so the first time
    // 0.00014ms to open the second time
    void *handle = dlopen("/home/samuel/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to open libtorch_cuda_cpp.so: " << dlerror() << std::endl;
        return 1;
    }
    dlclose(handle);
    return 0;
}

int open_and_run_entrypoint() {
    void *handle = dlopen(MYLIB_PATH, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to open mylib.so:" << dlerror() << std::endl;
        return 1;
    }
    void (*entry_point_execute)();
    entry_point_execute = (void (*)())dlsym(handle, "entry_point_execute");
    entry_point_execute();
    dlclose(handle);
    return 0;
}

int main() {
    std::string user_entry;
    while (user_entry != "e") {
        std::cout << "Press e to exit. Press p to proceed" << std::endl;
        std::cin >> user_entry;

        auto t0 = std::chrono::high_resolution_clock::now();
        open_large_dl();
        open_and_run_entrypoint();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_ms = t1 - t0;
        std::cout << "Took:" << time_ms.count() << "ms" << std::endl;
    }
    return 0;
}