#ifndef HELPER_H
#define HELPER_H
#include <cstdio>

/**
 * @brief Safely allocates memory, printing an error message if allocation fails.
 *
 * @param size The size of memory to allocate.
 * @return Pointer to allocated memory, or nullptr if allocation failed.
 */
inline void *malloc_s(const size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed\n");
    }
    return ptr;
}

/**
 * @brief Safely allocates contiguous memory, printing an error message if allocation fails.
 *
 * @param count The number of elements to allocate.
 * @param size The size of each element.
 * @return Pointer to allocated memory, or nullptr if allocation failed.
 */
inline void *calloc_s(const size_t count, const size_t size) {
    void *ptr = calloc(count, size);
    if (!ptr) {
        fprintf(stderr, "Contiguous allocation failed\n");
    }
    return ptr;
}

#endif
