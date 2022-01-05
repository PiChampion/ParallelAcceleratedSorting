#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <memory.h>

#include "graph.h"
#include "bfs.h"
#include "sort.h"
#include "edgelist.h"
#include "vertex.h"
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

int numThreads;

static void usage(void)
{
    printf("\nUsage: ./main -f <graph file> -r [root] -n [num threads]\n");
    printf("\t-f <graph file.txt>\n");
    printf("\t-h [help]\n");
    printf("\t-r [root/source]: BFS \n");
    printf("\t-n [num threads] default:max number of threads the system has\n");
    // _exit(-1);
}

#ifdef OPENMP_HARNESS
int main(int argc, char **argv)
{


    char *fvalue = NULL;
    char *rvalue = NULL;
    char *nvalue = NULL;

    int root = 0;

    numThreads = omp_get_max_threads();
    char *fnameb = NULL;

    int c;
    opterr = 0;

    while ((c = getopt (argc, argv, "f:r:n:h")) != -1)
    {
        switch (c)
        {
        case 'h':
            usage();
            break;
        case 'f':
            fvalue = optarg;
            fnameb = fvalue;
            break;
        case 'r':
            rvalue = optarg;
            root = atoi(rvalue);
            break;
            break;
        case 'n':
            nvalue = optarg;
            numThreads = atoi(nvalue);
            break;
        case '?':
            if (optopt == 'f')
                fprintf (stderr, "Option -%c <graph file> requires an argument  .\n", optopt);
            else if (optopt == 'r')
                fprintf (stderr, "Option -%c [root] requires an argument.\n", optopt);
            else if (optopt == 'n')
                fprintf (stderr, "Option -%c [num threads] requires an argument.\n", optopt);
            else if (isprint (optopt))
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf (stderr,
                         "Unknown option character `\\x%x'.\n",
                         optopt);
            usage();
            return 1;
        default:
            abort ();
        }
    }


    //Set number of threads for the program
    omp_set_nested(1);
    omp_set_num_threads(numThreads);

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "OPENMP Implementation");
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "File Name");
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", fnameb);
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Number of Threads");
    printf(" -----------------------------------------------------\n");
    printf("| %-51u | \n", numThreads);
    printf(" -----------------------------------------------------\n");


    struct Timer *timer = (struct Timer *) malloc(sizeof(struct Timer));


    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "New graph calculating size");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    struct Graph *graph = newGraph(fnameb);
    Stop(timer);
    printMessageWithtime("New Graph Created", Seconds(timer));


    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Populate Graph with edges");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    // populate the edge array from file
    loadEdgeArray(fnameb, graph);
    Stop(timer);
    printMessageWithtime("Time load edges to graph (Seconds)", Seconds(timer));
	
    // you need to parallelize this function
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "COUNT Sort Graph");
    printf(" -----------------------------------------------------\n");
    Start(timer);
	
	//graph = countSortEdgesBySource(graph);
    graph = radixSortEdgesBySourceOpenMP(graph); // you need to parallelize this function

    Stop(timer);
    printMessageWithtime("Time Sorting (Seconds)", Seconds(timer));


    // For testing purpose.

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Map vertices to Edges");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    mapVertices(graph);
    Stop(timer);
    printMessageWithtime("Time Mapping (Seconds)", Seconds(timer));

    printf(" *****************************************************\n");
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "BFS Algorithm (PUSH/PULL)");
    printf(" -----------------------------------------------------\n");

    printf("| %-51s | \n", "PUSH");

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "ROOT/SOURCE");
    printf(" -----------------------------------------------------\n");
    printf("| %-51u | \n", root);
    printf(" -----------------------------------------------------\n");
    Start(timer);

    breadthFirstSearchGraphPush(root, graph);

    Stop(timer);
    printMessageWithtime("Time BFS (Seconds)", Seconds(timer));

    Start(timer);
    freeGraph(graph);
    Stop(timer);
    printMessageWithtime("Free Graph (Seconds)", Seconds(timer));

    return 0;
}
#endif

#ifdef MPI_HARNESS
int main(int argc, char **argv)
{
    
	MPI_Init(&argc, &argv);

    char *fvalue = NULL;
    char *rvalue = NULL;
    char *nvalue = NULL;

    int root = 0;
	
	numThreads = omp_get_max_threads();

	int world_rank;			// Unique rank is assigned to each process in a communicator
	
	int world_size;			// Total number of ranks

	// Get this process' rank (process within a communicator)
	// MPI_COMM_WORLD is the default communicator
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get the total number ranks in this communicator
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
		char *fnameb = NULL;

		int c;
		opterr = 0;

		while ((c = getopt (argc, argv, "f:r:n:h")) != -1)
		{
			switch (c)
			{
			case 'h':
				if(world_rank == 0)
					usage();
				break;
			case 'f':
				fvalue = optarg;
				fnameb = fvalue;
				break;
			case 'r':
				rvalue = optarg;
				root = atoi(rvalue);
				break;
				break;
			case 'n':
				nvalue = optarg;
				numThreads = atoi(nvalue);
				break;
			case '?':
				if (optopt == 'f')
					fprintf (stderr, "Option -%c <graph file> requires an argument  .\n", optopt);
				else if (optopt == 'r')
					fprintf (stderr, "Option -%c [root] requires an argument.\n", optopt);
				else if (optopt == 'n')
					fprintf (stderr, "Option -%c [num threads] requires an argument.\n", optopt);
				else if (isprint (optopt))
					fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				else
					fprintf (stderr,
							 "Unknown option character `\\x%x'.\n",
							 optopt);
				usage();
				printf("\n\n\n WHAT IS THIS %d\n\n\n", world_rank);
				return 1;
			default:
				abort ();
			}
		}
		if(world_rank == 0) {
			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "MPI Implementation");


			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "File Name");
			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", fnameb);
			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "Number of Threads");
			printf(" -----------------------------------------------------\n");
			printf("| %-51u | \n", numThreads);
			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "Number of Nodes");
			printf(" -----------------------------------------------------\n");
			printf("| %-51u | \n", world_size);
			printf(" -----------------------------------------------------\n");
			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "New graph calculating size");
			printf(" -----------------------------------------------------\n");
		}
		struct Timer *timer = (struct Timer *) malloc(sizeof(struct Timer));
		Start(timer);
		struct Graph *graph = newGraph(fnameb);
		if(world_rank == 0) {
			Stop(timer);
			printMessageWithtime("New Graph Created", Seconds(timer));


			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "Populate Graph with edges");
			printf(" -----------------------------------------------------\n");
			Start(timer);
		}
		// populate the edge array from file
		loadEdgeArray(fnameb, graph);
		if(world_rank == 0) {
			Stop(timer);
			printMessageWithtime("Time load edges to graph (Seconds)", Seconds(timer));
			// you need to parallelize this function
			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "COUNT Sort Graph");
			printf(" -----------------------------------------------------\n");
		}
	////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
	////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	//printf("\nProcess %d beginning radix Sort\n", world_rank);
	Start(timer);
	graph = radixSortEdgesBySourceMPI(graph);		
	Stop(timer);
	//printf("\nProcess %d finished radix Sort\n", world_rank);
	////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
	////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	if(world_rank == 0) {
		printMessageWithtime("Time Sorting (Seconds)", Seconds(timer));
		// For testing purpose.
		printf(" -----------------------------------------------------\n");
		printf("| %-51s | \n", "Map vertices to Edges");
		printf(" -----------------------------------------------------\n");
		Start(timer);
		mapVertices(graph);
		Stop(timer);
		printMessageWithtime("Time Mapping (Seconds)", Seconds(timer));

		printf(" *****************************************************\n");
		printf(" -----------------------------------------------------\n");
		printf("| %-51s | \n", "BFS Algorithm (PUSH/PULL)");
		printf(" -----------------------------------------------------\n");

		printf("| %-51s | \n", "PUSH");

		printf(" -----------------------------------------------------\n");
		printf("| %-51s | \n", "ROOT/SOURCE");
		printf(" -----------------------------------------------------\n");
		printf("| %-51u | \n", root);
		printf(" -----------------------------------------------------\n");
		Start(timer);

		breadthFirstSearchGraphPush(root, graph);

		Stop(timer);
		printMessageWithtime("Time BFS (Seconds)", Seconds(timer));

		Start(timer);
		freeGraph(graph);
		Stop(timer);
		printMessageWithtime("Free Graph (Seconds)", Seconds(timer));
	}
	MPI_Finalize();
	
    return 0;
}
#endif

#ifdef HYBRID_HARNESS
int main(int argc, char **argv)
{
    
	MPI_Init(&argc, &argv);

    char *fvalue = NULL;
    char *rvalue = NULL;
    char *nvalue = NULL;

    int root = 0;
	
	numThreads = omp_get_max_threads();

	int world_rank;			// Unique rank is assigned to each process in a communicator
	
	int world_size;			// Total number of ranks

	// Get this process' rank (process within a communicator)
	// MPI_COMM_WORLD is the default communicator
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get the total number ranks in this communicator
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
		char *fnameb = NULL;

		int c;
		opterr = 0;

		while ((c = getopt (argc, argv, "f:r:n:h")) != -1)
		{
			switch (c)
			{
			case 'h':
				if(world_rank == 0)
					usage();
				break;
			case 'f':
				fvalue = optarg;
				fnameb = fvalue;
				break;
			case 'r':
				rvalue = optarg;
				root = atoi(rvalue);
				break;
				break;
			case 'n':
				nvalue = optarg;
				numThreads = atoi(nvalue);
				break;
			case '?':
				if (optopt == 'f')
					fprintf (stderr, "Option -%c <graph file> requires an argument  .\n", optopt);
				else if (optopt == 'r')
					fprintf (stderr, "Option -%c [root] requires an argument.\n", optopt);
				else if (optopt == 'n')
					fprintf (stderr, "Option -%c [num threads] requires an argument.\n", optopt);
				else if (isprint (optopt))
					fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				else
					fprintf (stderr,
							 "Unknown option character `\\x%x'.\n",
							 optopt);
				usage();
				printf("\n\n\n WHAT IS THIS %d\n\n\n", world_rank);
				return 1;
			default:
				abort ();
			}
		}
		
		//Set number of threads for the program
		omp_set_nested(1);
		omp_set_num_threads(numThreads);
		
		if(world_rank == 0) {
			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "MPI Implementation");


			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "File Name");
			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", fnameb);
			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "Number of Threads");
			printf(" -----------------------------------------------------\n");
			printf("| %-51u | \n", numThreads);
			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "Number of Nodes");
			printf(" -----------------------------------------------------\n");
			printf("| %-51u | \n", world_size);
			printf(" -----------------------------------------------------\n");
			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "New graph calculating size");
			printf(" -----------------------------------------------------\n");
		}
		struct Timer *timer = (struct Timer *) malloc(sizeof(struct Timer));
		Start(timer);
		struct Graph *graph = newGraph(fnameb);
		if(world_rank == 0) {
			Stop(timer);
			printMessageWithtime("New Graph Created", Seconds(timer));


			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "Populate Graph with edges");
			printf(" -----------------------------------------------------\n");
			Start(timer);
		}
		// populate the edge array from file
		loadEdgeArray(fnameb, graph);
		if(world_rank == 0) {
			Stop(timer);
			printMessageWithtime("Time load edges to graph (Seconds)", Seconds(timer));
			// you need to parallelize this function
			printf(" -----------------------------------------------------\n");
			printf("| %-51s | \n", "COUNT Sort Graph");
			printf(" -----------------------------------------------------\n");
		}
	////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
	////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	Start(timer);
	graph = radixSortEdgesBySourceHybrid(graph);		
	Stop(timer);
	////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
	////////////////////////////////////////////// BARRIER //////////////////////////////////////////////
	if(world_rank == 0) {
		printMessageWithtime("Time Sorting (Seconds)", Seconds(timer));

		// For testing purpose.
		printf(" -----------------------------------------------------\n");
		printf("| %-51s | \n", "Map vertices to Edges");
		printf(" -----------------------------------------------------\n");
		Start(timer);
		mapVertices(graph);
		Stop(timer);
		printMessageWithtime("Time Mapping (Seconds)", Seconds(timer));

		printf(" *****************************************************\n");
		printf(" -----------------------------------------------------\n");
		printf("| %-51s | \n", "BFS Algorithm (PUSH/PULL)");
		printf(" -----------------------------------------------------\n");

		printf("| %-51s | \n", "PUSH");

		printf(" -----------------------------------------------------\n");
		printf("| %-51s | \n", "ROOT/SOURCE");
		printf(" -----------------------------------------------------\n");
		printf("| %-51u | \n", root);
		printf(" -----------------------------------------------------\n");
		Start(timer);

		breadthFirstSearchGraphPush(root, graph);

		Stop(timer);
		printMessageWithtime("Time BFS (Seconds)", Seconds(timer));

		Start(timer);
		freeGraph(graph);
		Stop(timer);
		printMessageWithtime("Free Graph (Seconds)", Seconds(timer));
	}
	MPI_Finalize();
	
    return 0;
}
#endif



