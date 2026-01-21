#include <queue>
#include "dbscan.h"
#include "common.h"

/**
 * @brief Initialize the cluster labels to default 0.
 *
 * @param cluster Array of cluster labels.
 * @param n Number of points.
 */
static void init_cluster(int *cluster, const int n) {
    for (size_t i = 0; i < n; i++) {
        cluster[i] = NO_CLUSTER;
    }
}

static void free_adj_resources(int **cum_deg, int **adj, int **offset, int **degree) {
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
    int **cum_deg,
    int **adj,
    const double *x,
    const double *y,
    const int n,
    const double eps
) {
    int *degree = (int *) calloc_s(n, sizeof(int));
    if (!degree) return false;

    int total_neighbors = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (is_eps_neighbor(x[i], y[i], x[j], y[j], eps)) {
                degree[i]++;
                degree[j]++;
            }
        }
        total_neighbors += degree[i];
    }

    *cum_deg = (int *) malloc_s((n + 1) * sizeof(int));
    if (!*cum_deg) {
        free_adj_resources(cum_deg, nullptr, nullptr, &degree);
        return false;
    }

    (*cum_deg)[0] = 0;
    for (int i = 1; i <= n; i++) {
        (*cum_deg)[i] = (*cum_deg)[i - 1] + degree[i - 1];
    }

    *adj = (int *) malloc_s(total_neighbors * sizeof(int));
    int *offset = (int *) calloc_s(n, sizeof(int));
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

static void free_dbscan_resources(int **cum_deg, int **adj, int **queue) {
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
    int *cluster_count,
    const double *x,
    const double *y,
    const int n,
    const double eps,
    const int min_pts
) {
    int *cum_deg = nullptr;
    int *adj = nullptr;
    init_cluster(cluster, n);

    if (!compute_adj_list(&cum_deg, &adj, x, y, n, eps)) {
        fprintf(stderr, "Failed to compute adjacency list\n");
        free_dbscan_resources(&cum_deg, &adj, nullptr);
        return;
    }

    int cluster_id = NO_CLUSTER;
    int *queue = (int *) malloc_s(n * sizeof(int));
    if (!queue) {
        free_dbscan_resources(&cum_deg, &adj, &queue);
        return;
    }

    for (int p = 0; p < n; p++) {
        const int deg = cum_deg[p + 1] - cum_deg[p];

        if (cluster[p] != NO_CLUSTER || !is_core(deg, min_pts)) continue;

        // Assign cluster to core point
        cluster[p] = ++cluster_id;
        int head = 0, tail = 0;
        queue[tail++] = p;

        while (head < tail) {
            const int q = queue[head++];

            const int start = cum_deg[q];
            const int end = cum_deg[q + 1];
            const int q_deg = end - start;

            // Border points
            if (!is_core(q_deg, min_pts)) continue;

            for (int k = start; k < end; k++) {
                const int neighbor = adj[k];
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
