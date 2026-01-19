#ifndef HELPER_H
#define HELPER_H
#include <cstdio>
#include <cstdlib>


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
