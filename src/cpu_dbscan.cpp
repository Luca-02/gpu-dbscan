#include <queue>
#include "dbscan.h"
#include "helper.h"

/**
 * @brief Initialize the cluster labels to default 0.
 *
 * @param cluster Array of cluster labels.
 * @param n Number of points.
 */
static void init_clusters(int *cluster, const size_t n) {
    for (size_t i = 0; i < n; i++) {
        cluster[i] = NO_CLUSTER;
    }
}

static void free_adj_resources(size_t **cum_deg, size_t **adj, size_t **offset, size_t **degree) {
    if (cum_deg && *cum_deg) {
        free(*cum_deg);
        *cum_deg = nullptr;
    }
    if (adj && *adj) {
        free(*adj);
        *adj = nullptr;
    }
    if (offset && *offset) {
        free(*offset);
        *offset = nullptr;
    }
    if (*degree) {
        free(*degree);
        *degree = nullptr;
    }
}

static bool compute_adj_list(
    size_t **cum_deg,
    size_t **adj,
    const double *x,
    const double *y,
    const size_t n,
    const double eps
) {
    size_t *degree = (size_t *) calloc_s(n, sizeof(size_t));
    if (!degree) return false;

    size_t total_neighbors = 0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (is_eps_neighbor(x[i], y[i], x[j], y[j], eps)) {
                degree[i]++;
                degree[j]++;
            }
        }
        total_neighbors += degree[i];
    }

    *cum_deg = (size_t *) malloc_s((n + 1) * sizeof(size_t));
    if (!*cum_deg) {
        free_adj_resources(cum_deg, nullptr, nullptr, &degree);
        return false;
    }

    (*cum_deg)[0] = 0;
    for (size_t i = 1; i <= n; i++) {
        (*cum_deg)[i] = (*cum_deg)[i - 1] + degree[i - 1];
    }

    *adj = (size_t *) malloc_s(total_neighbors * sizeof(size_t));
    size_t *offset = (size_t *) calloc_s(n, sizeof(size_t));
    if (!*adj || !offset) {
        free_adj_resources(cum_deg, adj, &offset, &degree);
        return false;
    }

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (is_eps_neighbor(x[i], y[i], x[j], y[j], eps)) {
                (*adj)[(*cum_deg)[i] + offset[i]++] = j;
                (*adj)[(*cum_deg)[j] + offset[j]++] = i;
            }
        }
    }

    free_adj_resources(nullptr, nullptr, &offset, &degree);
    return true;
}

static void free_dbscan_resources(size_t **cum_deg, size_t **adj, size_t **queue) {
    if (*cum_deg) {
        free(*cum_deg);
        *cum_deg = nullptr;
    }
    if (*adj) {
        free(*adj);
        *adj = nullptr;
    }
    if (queue && *queue) {
        free(*queue);
        *queue = nullptr;
    }
}

void dbscan_cpu(
    int *cluster,
    size_t *cluster_count,
    const double *x,
    const double *y,
    const size_t n,
    const double eps,
    const size_t min_pts
) {
    size_t *cum_deg = nullptr;
    size_t *adj = nullptr;
    init_clusters(cluster, n);

    if (!compute_adj_list(&cum_deg, &adj, x, y, n, eps)) {
        fprintf(stderr, "Failed to compute adjacency list\n");
        free_dbscan_resources(&cum_deg, &adj, nullptr);
        return;
    }

    int cluster_id = NO_CLUSTER;
    size_t *queue = (size_t *) malloc_s(n * sizeof(size_t));
    if (!queue) {
        free_dbscan_resources(&cum_deg, &adj, &queue);
        return;
    }

    for (size_t p = 0; p < n; p++) {
        const size_t deg = cum_deg[p + 1] - cum_deg[p];

        if (cluster[p] != NO_CLUSTER || !is_core(deg, min_pts)) continue;

        // Assign cluster to core point
        cluster[p] = ++cluster_id;
        size_t head = 0, tail = 0;
        queue[tail++] = p;

        while (head < tail) {
            const size_t q = queue[head++];

            const size_t start = cum_deg[q];
            const size_t end = cum_deg[q + 1];
            const size_t q_deg = end - start;

            // Border points
            if (!is_core(q_deg, min_pts)) continue;

            for (size_t k = start; k < end; k++) {
                const size_t neighbor = adj[k];
                if (cluster[neighbor] == NO_CLUSTER) {
                    cluster[neighbor] = cluster_id;
                    queue[tail++] = neighbor;
                }
            }
        }
    }

    *cluster_count = cluster_id;

    free_dbscan_resources(&cum_deg, &adj, &queue);
}
