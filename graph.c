#include <stdio.h>
#include <stdlib.h>
#include <errno.h>


#include "graph.h"
#include "edgelist.h"
#include "vertex.h"


void   loadGraphInfo(const char * fname, struct Graph *graph){

		 loadEdgeArrayInfo(fname, &graph->num_vertices, &graph->num_edges);

}

void   printGraphInfo(struct Graph *graph, int edges){
	int i = 0;
	int max_edges = 0;
	
	if(edges < graph->num_edges) max_edges = edges;
	else max_edges = graph->num_edges;

	printf(" -----------------------------------------------------\n");
	printf(" --------------    GRAPH INFORMATION    --------------\n");
	printf(" -----------------------------------------------------\n");
	
	printf("EDGES: %d\n", graph->num_edges);
	printf("VERTICIES: %d\n", graph->num_vertices);
	
	printf(" --------------------    EDGES    --------------------\n");
	for(i = 0; i < max_edges; i++) {
		printf("%d -> %d\n", graph->sorted_edges_array[i].src, graph->sorted_edges_array[i].dest);
	}
	
/* 	printf(" -------------------    VERTICES    ------------------\n");
	for(i = 0; i < graph->num_vertices; i++) {
		
		printf("Vertex %d\n", graph->sorted_edges_array[graph->vertices[i].edges_idx].src);
	} */
}

// initialize a new graph from file
struct Graph * newGraph(const char * fname){

	int i;

	struct Graph* graph = (struct Graph*) malloc(sizeof(struct Graph));

	loadGraphInfo(fname, graph);

	graph->parents  = (int*) malloc( graph->num_vertices *sizeof(int));

    graph->vertices = newVertexArray(graph->num_vertices);
    graph->sorted_edges_array = newEdgeArray(graph->num_edges);

    for(i = 0; i < graph->num_vertices; i++){
        graph->parents[i] = -1;  
    }

    graph->iteration = 0;
    graph->processed_nodes = 0;

	return graph;
}


void freeGraph(struct Graph *graph){

    free(graph->vertices);
    free(graph->sorted_edges_array);

	free(graph);
}