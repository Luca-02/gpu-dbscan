#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H
#include <cstdio>

#define WARP_SIZE 32
#define DEFAULT_BLOCK_SIZE 128
#define BLOCK_SIZE_THRESHOLD 4096

/**
 * @brief Check for CUDA errors and exit if an error is found
 *
 * @param call The CUDA API call to check
 */
#define CUDA_CHECK(call)                                                                \
{                                                                                       \
    const cudaError_t error = call;                                                     \
    if (error != cudaSuccess) {                                                         \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(error) );                        \
        exit(EXIT_FAILURE);                                                             \
    }                                                                                   \
}

/**
 * @brief Function pointer type for calculating required shared memory for a given block size.
 *
 * @param blockSize The CUDA block size.
 * @return Required shared memory in bytes.
 */
typedef size_t (*smemFn)(uint32_t blockSize);

/**
 * @brief Compute zero shared memory for kernels that do not require dynamic shared memory.
 *
 * @return Always returns 0.
 */
inline size_t zeroSmem(uint32_t) {
    return 0;
}

/**
 * @brief Print device features.
 */
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

/**
 * @brief Computes the optimal kernel launch configuration (grid and block size)
 * optimized for Grid-Stride Loops.
 *
 * Instead of calculating a grid size for a 1:1 thread-to-element mapping (which
 * can result in massive grids and high scheduling overhead), this function
 * calculates a grid size sufficient to fully saturate the GPU's Multiprocessors (SMs),
 * but no larger.
 *
 * @note Kernels launched with this configuration must use a
 * Stride Loop pattern (e.g., `for (int i = tid; i < n; i += stride)`)
 * to process all elements.
 *
 * @tparam Kernel The CUDA kernel type.
 * @param gridSize Computed grid size.
 * @param blockSize Computed block size.
 * @param kernel The CUDA kernel to launch.
 * @param smemByteFn Function pointer to compute dynamic shared memory per block.
 * @param n Total number of elements to process.
 */
template<typename Kernel>
void getOptimalKernelSize(
    int *gridSize,
    int *blockSize,
    const Kernel kernel,
    const smemFn smemByteFn,
    const uint32_t n
) {
    if (n < BLOCK_SIZE_THRESHOLD) {
        // For very small workloads, use a fixed configuration
        *blockSize = DEFAULT_BLOCK_SIZE;
        *gridSize = (n + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
    }
    else {
        int minGridSize = 0;
        int optBlockSize = 0;

        // Calculate optimal block size for maximum theoretical occupancy
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
            &minGridSize,
            &optBlockSize,
            kernel,
            smemByteFn,
            0
        ));

        // Calculate blocks needed for a 1:1 mapping
        const int neededBlocks = (n + optBlockSize - 1) / optBlockSize;

        // Retrieve device properties to determine device saturation
        int device;
        int numSMs;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));

        int maxActiveBlocksPerSM;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocksPerSM,
            kernel,
            optBlockSize,
            smemByteFn(optBlockSize)
        ));

        // The saturation point is the number of SMs * max blocks the SM can handle.
        // This ensures we don't launch more blocks than the GPU can process in parallel.
        const int saturationBlock = numSMs * maxActiveBlocksPerSM;

        // Finalize outputs
        *blockSize = optBlockSize;

        // Use the saturation size, unless the input is so small it doesn't even fill the GPU once.
        *gridSize = min(neededBlocks, saturationBlock);
    }
}

/**
 * @brief Overload of getOptimalKernelSize for kernels that do not require dynamic shared memory.
 *
 * @tparam Kernel CUDA kernel type.
 * @param gridSize Computed grid size.
 * @param blockSize Computed block size.
 * @param kernel CUDA kernel to launch.
 * @param n Total number of elements to process.
 */
template<typename Kernel>
void getOptimalKernelSize(
    int *gridSize,
    int *blockSize,
    const Kernel kernel,
    const uint32_t n
) {
    getOptimalKernelSize(gridSize, blockSize, kernel, zeroSmem, n);
}

/**
 * @brief Launch a CUDA kernel with optional occupancy reporting and dynamic shared memory.
 * This version allows specifying the device properties and kernel name, which will
 * be used to print estimated occupancy.
 *
 * @tparam Kernel CUDA kernel type.
 * @tparam Args Variadic template parameters for kernel arguments.
 * @param prop CUDA device properties for occupancy calculation (optional).
 * @param kernelName Name of the kernel (used for logging occupancy).
 * @param kernel CUDA kernel to launch.
 * @param smemByteFn Function pointer to compute shared memory per block.
 * @param n Number of elements (used to calculate optimal block/grid).
 * @param args Arguments to pass to the kernel.
 */
template<typename Kernel, typename... Args>
void launchKernel(
    const cudaDeviceProp *prop,
    const char *kernelName,
    const Kernel kernel,
    const smemFn smemByteFn,
    const uint32_t n,
    Args... args
) {
    int gridSize = 0;
    int blockSize = 0;

    getOptimalKernelSize(&gridSize, &blockSize, kernel, smemByteFn, n);

    size_t smemBytes = smemByteFn ? smemByteFn(blockSize) : 0;

    kernel<<<gridSize, blockSize, smemBytes>>>(args...);
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

/**
 * @brief Launch a CUDA kernel without debugging information.
 * This version assumes dynamic shared memory is required and uses a shared memory
 * size computation function.
 *
 * @tparam Kernel CUDA kernel type.
 * @tparam Args Variadic template parameters for kernel arguments.
 * @param kernel CUDA kernel to launch.
 * @param smemFn Function pointer to compute shared memory per block.
 * @param n Number of elements.
 * @param args Arguments to pass to the kernel.
 */
template<typename Kernel, typename... Args>
void launchKernel(
    const Kernel kernel,
    const smemFn smemFn,
    const uint32_t n,
    Args... args
) {
    launchKernel(nullptr, nullptr, kernel, smemFn, n, args...);
}

/**
 * @brief Launch a CUDA kernel without debugging information and without dynamic shared memory.
 *
 * @tparam Kernel CUDA kernel type.
 * @tparam Args Variadic template parameters for kernel arguments.
 * @param kernel CUDA kernel to launch.
 * @param n Number of elements.
 * @param args Arguments to pass to the kernel.
 */
template<typename Kernel, typename... Args>
void launchKernel(
    const Kernel kernel,
    const uint32_t n,
    Args... args
) {
    launchKernel(nullptr, nullptr, kernel, zeroSmem, n, args...);
}

#endif
