#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <errno.h>
#include "timer.h"

#ifdef OPENMP_HARNESS
#include <omp.h>
#endif

#ifdef MPI_HARNESS
#include <mpi.h>
#endif

#ifdef HYBRID_HARNESS
#include <omp.h>
#include <mpi.h>
#endif

#include "sort.h"
#include "graph.h"

void printMessageWithtime(const char *msg, double time)
{

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", msg);
    printf(" -----------------------------------------------------\n");
    printf("| %-51f | \n", time);
    printf(" -----------------------------------------------------\n");

}

struct Graph *countSortEdgesBySource (struct Graph *graph)
{

    int i;
    int key;
    int pos;
    struct Edge *sorted_edges_array = newEdgeArray(graph->num_edges);

    // auxiliary arrays, allocated at the start up of the program
    int *vertex_count = (int *)malloc(graph->num_vertices * sizeof(int)); // needed for Counting Sort

    for(i = 0; i < graph->num_vertices; ++i)
    {
        vertex_count[i] = 0;
    }

    // count occurrence of key: id of a source vertex
    for(i = 0; i < graph->num_edges; ++i)
    {
        key = graph->sorted_edges_array[i].src;
        vertex_count[key]++;
    }

    // transform to cumulative sum
    for(i = 1; i < graph->num_vertices; ++i)
    {
        vertex_count[i] += vertex_count[i - 1];
    }

    // fill-in the sorted array of edges
    for(i = graph->num_edges - 1; i >= 0; --i)
    {
        key = graph->sorted_edges_array[i].src;
        pos = vertex_count[key] - 1;
        sorted_edges_array[pos] = graph->sorted_edges_array[i];
        vertex_count[key]--;
    }



    free(vertex_count);
    free(graph->sorted_edges_array);

    graph->sorted_edges_array = sorted_edges_array;

    return graph;

}


#ifdef OPENMP_HARNESS
struct Graph* countSortEdgesBySourceOpenMP (struct Graph* graph, int radix){

	int serial_i;
	int base = 0;
    struct Edge *sorted_edges_array = newEdgeArray(graph->num_edges);
	
    // auxiliary arrays, allocated at the start up of the program
	int **vertex_count = (int **)malloc(numThreads * sizeof(int *));
    for (serial_i = 0; serial_i < numThreads; serial_i++)
		vertex_count[serial_i] = (int *)malloc(RADIX_VERTICIES * sizeof(int));
	
////////////////////////////////////////////// Start of parallel region //////////////////////////////////////////////
#pragma omp parallel
{
	int i, j;
    int key;
    int pos;
	int tid = omp_get_thread_num();
	
	int start = tid * (graph->num_edges / numThreads) + ((tid <= (graph->num_edges % numThreads)) ? tid : (graph->num_edges % numThreads));
	int end = ((tid + 1) * (graph->num_edges / numThreads) + (((tid + 1) <= (graph->num_edges % numThreads)) ? (tid + 1) : (graph->num_edges % numThreads))) - 1;
	
    for(i = 0; i < RADIX_VERTICIES; ++i) {
        vertex_count[tid][i] = 0;
    }

    // count occurrence of key: id of a source vertex
    for(i = start; i <= end; ++i) {
        key = (graph->sorted_edges_array[i].src >> (radix * RADIX_BITS)) & RADIX_MASK;
        vertex_count[tid][key]++;
    }

#pragma omp barrier
if(tid == 0)
{
	printf("%d\n", omp_get_num_threads());
    // transform to cumulative sum
    for(i = 0; i < RADIX_VERTICIES; ++i) {
		for(j = 0; j < numThreads; j++) {
			base = base + vertex_count[j][i];
			vertex_count[j][i] = base;
		}
    }
}
#pragma omp barrier
    // fill-in the sorted array of edges
    for(i = end; i >= start; --i) {
        key = (graph->sorted_edges_array[i].src >> (radix * RADIX_BITS)) & RADIX_MASK;
        pos = vertex_count[tid][key] - 1;
        sorted_edges_array[pos] = graph->sorted_edges_array[i];
        vertex_count[tid][key]--;
    }
}

/////////////////////////////////////////////// End of parallel region ///////////////////////////////////////////////
	for (serial_i = 0; serial_i < numThreads; serial_i++) {
		free(vertex_count[serial_i]);
	}

	free(graph->sorted_edges_array);
    graph->sorted_edges_array = sorted_edges_array;

    return graph;

}

struct Graph *radixSortEdgesBySourceOpenMP (struct Graph *graph)
{
	int i;
	
    printf("*** START Radix Sort Edges By Source OpenMP *** \n");
	for(i = 0; i < RADIX_ITERATIONS; i++) {
        graph = countSortEdgesBySourceOpenMP(graph, i);
    }
    return graph;
}
#endif

#ifdef MPI_HARNESS

/* our reduction operation */
void sum_struct_ts(void *in, void *inout, int *len, MPI_Datatype *type){
    /* ignore type, just trust that it's our struct type */
	int i=0;
	
    struct Edge *invals    = in;
    struct Edge *inoutvals = inout;

    for (i=0; i<*len; i++) {
        inoutvals[i].src  += invals[i].src;
        inoutvals[i].dest  += invals[i].dest;
    }

    return;
}

void defineStruct(MPI_Datatype *tstype) {
    const int count = 2;
    int          blocklens[count];
    MPI_Datatype types[count];
    MPI_Aint     disps[count];
	int i=0;
	
    for (i=0; i < count; i++) {
        types[i] = MPI_INT;
        blocklens[i] = 1;
    }

    disps[0] = offsetof(struct Edge,src);
    disps[1] = offsetof(struct Edge,dest);

    MPI_Type_create_struct(count, blocklens, disps, types, tstype);
    MPI_Type_commit(tstype);
}

struct Graph* countSortEdgesBySourceMPI (struct Graph* graph, int radix){

	int base = 0;
    struct Edge *sorted_edges_array = newEdgeArray(graph->num_edges);
	struct Edge *global_sorted_edges_array = newEdgeArray(graph->num_edges);
	
	
	MPI_Datatype structtype;
    MPI_Op       sumstruct;
	defineStruct(&structtype);
	MPI_Op_create(sum_struct_ts, 1, &sumstruct);
	
////////////////////////////////////////////// Start of parallel region //////////////////////////////////////////////
	int i, j, k;
    int key;
    int pos;
	
	int world_rank;			// Unique rank is assigned to each process in a communicator
	
	int world_size;			// Total number of ranks
	
	struct Timer *myTimer = (struct Timer *) malloc(sizeof(struct Timer));

	// Get this process' rank (process within a communicator)
	// MPI_COMM_WORLD is the default communicator
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get the total number ranks in this communicator
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
    // auxiliary arrays, allocated at the start up of the program
	int *vertex_count = (int *)malloc(RADIX_VERTICIES * sizeof(int)); // needed for Counting Sort
	
	int** buffer = (int **)malloc(world_size * sizeof(int *));
	
	if(world_rank == 0) {
		// auxiliary arrays, allocated at the start up of the program
		for (i = 0; i < world_size; i++)
			buffer[i] = (int *)malloc(RADIX_VERTICIES * sizeof(int));
	}
	
	int start = world_rank * (graph->num_edges / world_size) + ((world_rank <= (graph->num_edges % world_size)) ? world_rank : (graph->num_edges % world_size));
	int end = ((world_rank + 1) * (graph->num_edges / world_size) + (((world_rank + 1) <= (graph->num_edges % world_size)) ? (world_rank + 1) : (graph->num_edges % world_size))) - 1;
	
	//printf("\nMPI Process %d Zeroing Vertex_count\n", world_rank);
    for(i = 0; i < RADIX_VERTICIES; ++i) {
        vertex_count[i] = 0;
    }

	//printf("\nMPI Process %d Counting Occurances into Vertex_count\n", world_rank);
    // count occurrence of key: id of a source vertex
    for(i = start; i <= end; ++i) {
        key = (graph->sorted_edges_array[i].src >> (radix * RADIX_BITS)) & RADIX_MASK;
        vertex_count[key]++;
    }
	
////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	
	if (world_rank == 0) {
		Start(myTimer);
		for(j = 1; j < world_size; j++) {
				MPI_Recv(buffer[j], RADIX_VERTICIES, MPI_INT, j, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//printf("\nMPI Process %d Received buffer [", world_rank);
				//for(k = 0; k < RADIX_VERTICIES; k++) {
				//	printf("%d, ", buffer[j][k]);
				//}
				//printf("] from process %d\n", j);
		}
		Stop(myTimer);
		printMessageWithtime("Time spend receiving buffers from other threads", Seconds(myTimer));
		// transform to cumulative sum
		for(i = 0; i < RADIX_VERTICIES; ++i) {
			for(j = 0; j < world_size; j++) {
				if(j == 0) {
					base = base + vertex_count[i];
					vertex_count[i] = base;
				}
				else {
					base = base + buffer[j][i];
					buffer[j][i] = base;
				}
			}
		}
		Start(myTimer);
		for(j = 1; j < world_size; j++) {
			MPI_Send(buffer[j], RADIX_VERTICIES, MPI_INT, j, world_rank, MPI_COMM_WORLD);
			//printf("\nMPI Process %d Sent buffer to process %d\n", world_rank, j);
		}
		Stop(myTimer);
		printMessageWithtime("Time spend sending back buffers to other threads", Seconds(myTimer));
	}
	else {
		MPI_Send(vertex_count, RADIX_VERTICIES, MPI_INT, 0, world_rank, MPI_COMM_WORLD);
		//printf("\nMPI Process %d Sent buffer to process %d\n", world_rank, 0);
		MPI_Recv(vertex_count, RADIX_VERTICIES, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//printf("\nMPI Process %d Received buffer to process %d\n", world_rank, 0);
	}

////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
////////////////////////////////////////////// BARRIER //////////////////////////////////////////////

	//printf("\nMPI Process %d filling local sorted array\n", world_rank);
    // fill-in the sorted array of edges
    for(i = end; i >= start; --i) {
        key = (graph->sorted_edges_array[i].src >> (radix * RADIX_BITS)) & RADIX_MASK;
        pos = vertex_count[key] - 1;
        sorted_edges_array[pos] = graph->sorted_edges_array[i];
        vertex_count[key]--;
    }
	
	//printf("\nMPI Process %d participating in AllREDUCE\n", world_rank);
	if (world_rank == 0) Start(myTimer);
	MPI_Allreduce(sorted_edges_array, global_sorted_edges_array, graph->num_edges, structtype, sumstruct, MPI_COMM_WORLD);
	if (world_rank == 0) {
		Stop(myTimer);
		printMessageWithtime("Time spend reducing edge arrays", Seconds(myTimer));
	}
	////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
	////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	
	free(vertex_count);
	
	if (world_rank == 0) {
		for (i = 0; i < world_size; i++) {
			free(buffer[i]);
		}
	}
	else {
		free(buffer);
	}
	
	free(sorted_edges_array);
	free(myTimer);
	
	free(graph->sorted_edges_array);
	graph->sorted_edges_array = global_sorted_edges_array;
    
	return graph;

}

struct Graph *radixSortEdgesBySourceMPI (struct Graph *graph)
{
    
    int i;

    printf("*** START Radix Sort Edges By Source MPI *** \n");
    for(i = 0; i < RADIX_ITERATIONS; i++) {
        graph = countSortEdgesBySourceMPI(graph, i);
    }

    return graph;
}
#endif

#ifdef HYBRID_HARNESS
/* our reduction operation */
void sum_struct_ts(void *in, void *inout, int *len, MPI_Datatype *type){
    /* ignore type, just trust that it's our struct type */
	int i=0;
	
    struct Edge *invals    = in;
    struct Edge *inoutvals = inout;

    for (i=0; i<*len; i++) {
        inoutvals[i].src  += invals[i].src;
        inoutvals[i].dest  += invals[i].dest;
    }

    return;
}

void defineStruct(MPI_Datatype *tstype) {
    const int count = 2;
    int          blocklens[count];
    MPI_Datatype types[count];
    MPI_Aint     disps[count];
	int i=0;
	
    for (i=0; i < count; i++) {
        types[i] = MPI_INT;
        blocklens[i] = 1;
    }

    disps[0] = offsetof(struct Edge,src);
    disps[1] = offsetof(struct Edge,dest);

    MPI_Type_create_struct(count, blocklens, disps, types, tstype);
    MPI_Type_commit(tstype);
}

struct Graph* countSortEdgesBySourceHybrid (struct Graph* graph, int radix){

	int base = 0;
    struct Edge *sorted_edges_array = newEdgeArray(graph->num_edges);
	struct Edge *global_sorted_edges_array = newEdgeArray(graph->num_edges);
	int serial_i, serial_j;
	
	
	MPI_Datatype structtype;
    MPI_Op       sumstruct;
	defineStruct(&structtype);
	MPI_Op_create(sum_struct_ts, 1, &sumstruct);
	
////////////////////////////////////////////// Start of parallel region //////////////////////////////////////////////
	int world_rank;			// Unique rank is assigned to each process in a communicator
	
	int world_size;			// Total number of ranks
	
	struct Timer *myTimer = (struct Timer *) malloc(sizeof(struct Timer));

	// Get this process' rank (process within a communicator)
	// MPI_COMM_WORLD is the default communicator
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get the total number ranks in this communicator
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	int *vertex_count = (int *)malloc(numThreads * RADIX_VERTICIES * sizeof(int));
	
	int* buffer = (int *)malloc(world_size * numThreads * RADIX_VERTICIES * sizeof(int));
	
	int start = world_rank * (graph->num_edges / world_size) + ((world_rank <= (graph->num_edges % world_size)) ? world_rank : (graph->num_edges % world_size));
	int end = ((world_rank + 1) * (graph->num_edges / world_size) + (((world_rank + 1) <= (graph->num_edges % world_size)) ? (world_rank + 1) : (graph->num_edges % world_size))) - 1;

	int s_size = end - start + 1;

#pragma omp parallel
{
	int i, j, k;
    int key;
    int pos;
	int tid = omp_get_thread_num();
	
	int thread_start = start + tid * (s_size / numThreads) + ((tid <= (s_size % numThreads)) ? tid : (s_size % numThreads));
	int thread_end = start + ((tid + 1) * (s_size / numThreads) + (((tid + 1) <= (s_size % numThreads)) ? (tid + 1) : (s_size % numThreads))) - 1;
	
	for(i = 0; i < RADIX_VERTICIES; ++i) {
		vertex_count[tid * RADIX_VERTICIES + i] = 0;
	}

    // count occurrence of key: id of a source vertex
    for(i = thread_start; i <= thread_end; ++i) {
        key = (graph->sorted_edges_array[i].src >> (radix * RADIX_BITS)) & RADIX_MASK;
        vertex_count[tid * RADIX_VERTICIES + key]++;
    }
	
////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
#pragma omp barrier
	if(tid == 0) {
		//printf("process %d gathering buffer\n", world_rank);
		if(world_size > 1) MPI_Gather(vertex_count, numThreads * RADIX_VERTICIES, MPI_INT, buffer, numThreads * RADIX_VERTICIES, MPI_INT, 0, MPI_COMM_WORLD);
		if (world_rank == 0) {
			//printf("\nMPI Process %d holds buffer ", world_rank);
			//for(j = 0; j < world_size; j++) {
			//	printf("[");
			//	for(k = 0; k < RADIX_VERTICIES; k++) {
			//		printf("%d, ", buffer[numThreads*RADIX_VERTICIES*j + k]);
			//	}
			//	printf("]\n");
			//}
			//MPI_Gather(vertex_count, numThreads * RADIX_VERTICIES, MPI_INT, buffer, numThreads * RADIX_VERTICIES, MPI_INT, 0, MPI_COMM_WORLD);
			//for(j = 1; j < world_size; j++) {
			//		MPI_Recv(&(buffer[numThreads*RADIX_VERTICIES*j]), RADIX_VERTICIES*numThreads, MPI_INT, j, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			//}
			// transform to cumulative sum
			for(i = 0; i < RADIX_VERTICIES; ++i) {
				for(j = 0; j < world_size; j++) {
					for(k = 0; k < numThreads; k++) {
						//if(j == 0) {
						//	base = base + vertex_count[k*RADIX_VERTICIES + i];
						//	vertex_count[k*RADIX_VERTICIES + i] = base;
						//}
						//else {
							if(world_size > 1) {
								base = base + buffer[j*numThreads*RADIX_VERTICIES + k * RADIX_VERTICIES+ i];
								buffer[j*numThreads*RADIX_VERTICIES + k * RADIX_VERTICIES+ i] = base;
							}
							else {
								base = base + vertex_count[k*RADIX_VERTICIES + i];
								vertex_count[k*RADIX_VERTICIES + i] = base;
							}
						//}
					}
				}
			}
			//MPI_Scatter(buffer, numThreads * RADIX_VERTICIES, MPI_INT, vertex_count, numThreads * RADIX_VERTICIES, MPI_INT, 0, MPI_COMM_WORLD);
			//for(j = 1; j < world_size; j++) {
			//	MPI_Send(&(buffer[numThreads*RADIX_VERTICIES*j]), RADIX_VERTICIES*numThreads, MPI_INT, j, world_rank, MPI_COMM_WORLD);
			//}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		//printf("process %d scattering buffer\n", world_rank);
		if(world_size > 1) MPI_Scatter(buffer, numThreads * RADIX_VERTICIES, MPI_INT, vertex_count, numThreads * RADIX_VERTICIES, MPI_INT, 0, MPI_COMM_WORLD);
		//printf("\nMPI Process %d holds vertex ", world_rank);
		//printf("[");
		//for(k = 0; k < RADIX_VERTICIES; k++) {
		//	printf("%d, ", vertex_count[k]);
		//}
		//printf("]\n");
		
		//else {
		//	MPI_Send(vertex_count, RADIX_VERTICIES*numThreads, MPI_INT, 0, world_rank, MPI_COMM_WORLD);
		//	MPI_Recv(vertex_count, RADIX_VERTICIES*numThreads, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//}
	}
#pragma omp barrier
////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
    // fill-in the sorted array of edges
    for(i = thread_end; i >= thread_start; --i) {
        key = (graph->sorted_edges_array[i].src >> (radix * RADIX_BITS)) & RADIX_MASK;
        pos = vertex_count[tid * RADIX_VERTICIES + key] - 1;
        sorted_edges_array[pos] = graph->sorted_edges_array[i];
        vertex_count[tid * RADIX_VERTICIES + key]--;
    }
	#pragma omp barrier
	if(tid == 0) {
		if (world_rank == 0) Start(myTimer);
		MPI_Allreduce(sorted_edges_array, global_sorted_edges_array, graph->num_edges, structtype, sumstruct, MPI_COMM_WORLD);
		if (world_rank == 0) {
			Stop(myTimer);
			printMessageWithtime("Time spend reducing edge arrays", Seconds(myTimer));
		}
	}
}
	////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
	////////////////////////////////////////////// BARRIER //////////////////////////////////////////////

	//printf("process %d freeing vertex\n", world_rank);
	free(vertex_count);
	//printf("process %d freeing timer\n", world_rank);
	free(myTimer);
	//printf("process %d freeing buffer\n", world_rank);
	free(buffer);
	//printf("process %d freeing local sorted edges\n", world_rank);	
	free(sorted_edges_array);
	
	///printf("process %d freeing graph edges\n", world_rank);
	free(graph->sorted_edges_array);
	graph->sorted_edges_array = global_sorted_edges_array;
    
	//printf("process %d returning graph\n", world_rank);
	return graph;

}

struct Graph *radixSortEdgesBySourceHybrid (struct Graph *graph)
{
    int i;

    printf("*** START Radix Sort Edges By Source Hybrid *** \n");
    for(i = 0; i < RADIX_ITERATIONS; i++) {
        graph = countSortEdgesBySourceHybrid(graph, i);
    }

    return graph;
}
#endif
