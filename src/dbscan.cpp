#include <cstdio>
#include <queue>
#include "dbscan.h"

void free_graph(Graph *graph) {
    free(graph->adj);
    free(graph->cum_deg);
    free(graph);
}

void build_graph(Graph *graph, const Point *points, const int n, const double eps) {
    graph->n = n;
    graph->cum_deg = (int *) malloc((n + 1) * sizeof(int));

    int *degree = (int *) calloc(n, sizeof(int));
    int total_neighbors = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && distance(&points[i], &points[j]) <= eps) {
                degree[i]++;
            }
        }
        total_neighbors += degree[i];
    }

    graph->cum_deg[0] = 0;
    for (int i = 1; i <= n; i++) {
        graph->cum_deg[i] = graph->cum_deg[i - 1] + degree[i - 1];
    }

    graph->adj = (int *) malloc(total_neighbors * sizeof(int));
    int *offset = (int *) calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && distance(&points[i], &points[j]) <= eps) {
                const int pos = graph->cum_deg[i] + offset[i];
                graph->adj[pos] = j;
                offset[i]++;
            }
        }
    }

    free(degree);
    free(offset);
}

void dbscan(Point *points, const int n, const double eps, const int min_pts) {
    Graph *graph = (Graph *) malloc(sizeof(Graph));
    build_graph(graph, points, n, eps);

    int cluster_id = 0;
    std::queue<int> queue;

    for (int i = 0; i < n; i++) {
        if (points[i].label != UNDEFINED) continue;

        const int deg = graph->cum_deg[i + 1] - graph->cum_deg[i];

        // +1 because the point itself is also included
        if (deg + 1 < min_pts) {
            points[i].label = NOISE;
            continue;
        }

        // Assign cluster to core point
        points[i].label = cluster_id;
        queue.push(i);

        while (!queue.empty()) {
            const int p = queue.front();
            queue.pop();

            const int start = graph->cum_deg[p];
            const int end = graph->cum_deg[p + 1];
            const int p_deg = end - start;

            if (p_deg + 1 < min_pts) continue;

            for (int k = start; k < end; k++) {
                int neighbor = graph->adj[k];

                if (points[neighbor].label == UNDEFINED) {
                    points[neighbor].label = cluster_id;
                    queue.push(neighbor);
                } else if (points[neighbor].label == NOISE) {
                    // Border point: from noise to cluster
                    points[neighbor].label = cluster_id;
                }
            }
        }

        cluster_id++;
    }

    free_graph(graph);
}

void write_output(const char *filename, const Point *points, const int n) {
    FILE *file = fopen(filename, "w");
    if (!file) return;

    for (int i = 0; i < n; i++) {
        fprintf(
            file,
            "%lf %lf %d\n",
            points[i].x,
            points[i].y,
            points[i].label
        );
    }

    fclose(file);
}
