#ifndef SORT_H
#define SORT_H

#include "edgelist.h"
#include "graph.h"
// Order edges by id of a source vertex,
// using the Counting Sort
// Complexity: O(E + V)

#define RADIX_BITS					8
#define RADIX_VERTICIES				256
#define RADIX_MASK                  0xff
#define RADIX_ITERATIONS            4

void printMessageWithtime(const char *msg, double time);

#ifdef OPENMP_HARNESS
struct Graph *radixSortEdgesBySourceOpenMP (struct Graph *graph);
struct Graph *countSortEdgesBySourceOpenMP (struct Graph *graph, int radix);
#endif

#ifdef MPI_HARNESS
struct Graph *radixSortEdgesBySourceMPI (struct Graph *graph);
struct Graph *countSortEdgesBySourceMPI (struct Graph *graph, int radix);
#endif

#ifdef HYBRID_HARNESS
struct Graph *radixSortEdgesBySourceHybrid (struct Graph *graph);
struct Graph *countSortEdgesBySourceHybrid (struct Graph *graph, int radix);
#endif

struct Graph *countSortEdgesBySource (struct Graph *graph);

extern int numThreads;

#endif