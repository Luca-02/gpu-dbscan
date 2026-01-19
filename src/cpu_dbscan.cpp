#include <queue>
#include "dbscan.h"


void compute_adj_list(int **cum_deg, int **adj, const double *x, const double *y, const int n, const double eps) {
    *cum_deg = (int *) malloc((n + 1) * sizeof(int));

    int *degree = (int *) calloc(n, sizeof(int));
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

    (*cum_deg)[0] = 0;
    for (int i = 1; i <= n; i++) {
        (*cum_deg)[i] = (*cum_deg)[i - 1] + degree[i - 1];
    }

    *adj = (int *) malloc(total_neighbors * sizeof(int));
    int *offset = (int *) calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (is_eps_neighbor(x[i], y[i], x[j], y[j], eps)) {
                (*adj)[(*cum_deg)[i] + offset[i]++] = j;
                (*adj)[(*cum_deg)[j] + offset[j]++] = i;
            }
        }
    }

    free(degree);
    free(offset);
}

void dbscan_cpu(int *cluster, const double *x, const double *y, const int n, const double eps, const int min_pts) {
    for (int i = 0; i < n; i++) {
        cluster[i] = UNDEFINED;
    }

    int *cum_deg, *adj;
    compute_adj_list(&cum_deg, &adj, x, y, n, eps);

    int cluster_id = 0;
    int *queue = (int *) malloc(n * sizeof(int));

    for (int p = 0; p < n; p++) {
        if (cluster[p] != UNDEFINED) continue;

        const int deg = cum_deg[p + 1] - cum_deg[p];

        if (!is_core(deg, min_pts)) {
            cluster[p] = NOISE;
            continue;
        }

        // Assign cluster to core point
        cluster[p] = cluster_id;
        int head = 0;
        int tail = 0;
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
                if (cluster[neighbor] == UNDEFINED || cluster[neighbor] == NOISE) {
                    cluster[neighbor] = cluster_id;
                    queue[tail++] = neighbor;
                }
            }
        }

        cluster_id++;
    }

    free(cum_deg);
    free(adj);
    free(queue);
}
