#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H
#include <cstdio>

#define WARP_SIZE 32

#define CUDA_CHECK(call)                                                                \
{                                                                                       \
    const cudaError_t error = call;                                                     \
    if (error != cudaSuccess) {                                                         \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(error) );                        \
        exit(EXIT_FAILURE);                                                             \
    }                                                                                   \
}

typedef size_t (*smemFn)(uint32_t blockSize);

inline size_t zeroSmem(uint32_t) {
    return 0;
}

inline void deviceFeat() {
    int dev = 0;
    int driverVersion = 0, runtimeVersion = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    int maxWarps = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;

    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    printf("------------------------------------------------------------\n");
    printf("  CUDA Capability Major/Minor version number:   %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total amount of global memory:                %.2f MB\n", (float) deviceProp.totalGlobalMem / 1048576.0f);
    printf("  Total amount of shared memory per block       %.2f KB\n", (float) deviceProp.sharedMemPerBlock / 1024.0f);
    printf("  Total shared memory per multiprocessor:       %.2f KB\n",
           (float) deviceProp.sharedMemPerMultiprocessor / 1024.0f);
    printf("  Total number of registers available per block %d\n", deviceProp.regsPerBlock);
    printf("  Maximum number of threads per multiprocessor  %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of warps per multiprocessor    %d\n", maxWarps);
    printf("------------------------------------------------------------\n\n");
}

template<typename Kernel, typename... Args>
void launchKernel(
    const cudaDeviceProp *prop,
    const char *kernelName,
    const Kernel kernel,
    const smemFn smemByteFn,
    const uint32_t n,
    Args... args
) {
    int minGridSize = 0;
    int blockSize = 0;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &minGridSize, &blockSize, kernel, smemByteFn, n
    ));

    int gridSize = (n + blockSize - 1) / blockSize;
    size_t smemBytes = smemByteFn ? smemByteFn(blockSize) : 0;

    kernel<<<gridSize, blockSize, smemBytes>>>(args...);
    // kernel<<<minGridSize, blockSize, smemBytes>>>(args...);
    CUDA_CHECK(cudaGetLastError());

    if (prop && kernelName) {
        int numBlocks;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks, kernel, blockSize, smemBytes
        ));

        const int activeWarps = numBlocks * blockSize / prop->warpSize;
        const int maxWarps = prop->maxThreadsPerMultiProcessor / prop->warpSize;
        const float occupancy = 100.0 * activeWarps / maxWarps;

        printf("%s occupancy [blockSize = %d, activeWarps = %d]: %2.2f%%\n",
               kernelName, blockSize, activeWarps, occupancy);
    }
}

template<typename Kernel, typename... Args>
void launchKernel(
    const Kernel kernel,
    const smemFn smemFn,
    const uint32_t n,
    Args... args
) {
    launchKernel(nullptr, nullptr, kernel, smemFn, n, args...);
}

template<typename Kernel, typename... Args>
void launchKernel(
    const Kernel kernel,
    const uint32_t n,
    Args... args
) {
    launchKernel(nullptr, nullptr, kernel, zeroSmem, n, args...);
}

#endif
