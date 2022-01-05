#ifndef EDGELIST_H
#define EDGELIST_H

#include "graph.h"

typedef struct Edge{
    int src;   	  // id of a source vertex
    int dest;     // id of a destination vertex
} E;

struct Edge * newEdgeArray(int numOfEdges);
struct Graph * loadEdgeArray(const char * fname, struct Graph *graph);
void   loadEdgeArrayInfo(const char * fname, int *numOfVertices, int *numOfEdges);
int maxTwoIntegers(int num1, int num2);
void printEdgeArray(struct Edge *edgeArray, int numOfEdges);

#endif