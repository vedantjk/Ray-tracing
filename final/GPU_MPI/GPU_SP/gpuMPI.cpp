#include <mpi.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
#include "gpuMPI.h"

int main(int argc, char *argv[])
{
    int rank, size;
    if (argc != 5)
    {
        printf("Usage: %s <nrays> <ngrid> <num_blocks> <threads_per_block> \n", argv[0]);
        return 1;
    }
    int Nrays = atoi(argv[1]);
    int n = atoi(argv[2]);
    int threadsPerBlock = atoi(argv[4]);
    int numBlocks = atoi(argv[3]);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("%d\n", size);

    float *G = (float *)malloc(sizeof(float) * (n + 1) * (n + 1));
    float *localG = (float *)malloc(sizeof(float) * (n + 1) * (n + 1));

    computeGPU(localG, Nrays / size, n, threadsPerBlock, numBlocks);

    // Reduce localG data across all processes
    MPI_Reduce(localG, G, (n + 1) * (n + 1), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        FILE *outputFile = fopen("output.txt", "w");
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                fprintf(outputFile, "%f ", G[i * (n + 1) + j]);
            }
            fprintf(outputFile, "\n");
        }
        fclose(outputFile);
    }
    free(G);
    free(localG);
    MPI_Finalize();
    return 0;
}

// Shut down MPI cleanly if something goes wrong
void my_abort(int err)
{
    cout << "Test FAILED\n";
    MPI_Abort(MPI_COMM_WORLD, err);
}
