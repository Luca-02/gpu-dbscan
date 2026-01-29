#ifndef HELPER_H
#define HELPER_H
#include <cstdio>

/**
 * @brief Compares two strings for sorting.
 *
 * @param a Pointer to first string.
 * @param b Pointer to second string.
 * @return Negative if a < b, positive if a > b, zero if equal.
 */
inline int compareStrings(const void *a, const void *b) {
    const char *s1 = *(char **) a;
    const char *s2 = *(char **) b;
    return strcmp(s1, s2);
}

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
