#ifndef HELPER_H
#define HELPER_H
#include <cstdio>
#include <cstdlib>

#define X_INDEX(i) (2 * (i))
#define Y_INDEX(i) (2 * (i) + 1)

#define CUDA_CHECK(call)                                                                \
{                                                                                       \
    const cudaError_t error = call;                                                     \
    if (error != cudaSuccess) {                                                         \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                          \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));    \
    }                                                                                   \
}

inline void *malloc_s(const size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed\n");
    }
    return ptr;
}

inline void *calloc_s(const size_t count, const size_t size) {
    void *ptr = calloc(count, size);
    if (!ptr) {
        fprintf(stderr, "Contiguous allocation failed\n");
    }
    return ptr;
}

#endif
