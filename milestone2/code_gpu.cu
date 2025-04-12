#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
// Elapsed Time: 544.907288 ms with 1000 80
// total: 0.939 s
#define n 1000         // Grid dimensions
#define Wmax 2.0       // Window bounds
#define Nrays (int)1e9 // Number of rays

typedef struct
{
    double x, y, z;
} Vector;

typedef struct
{
    int i, j;
} GridPoint;

__device__ double dotProduct(Vector v1, Vector v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ Vector subtract(Vector v1, Vector v2)
{
    Vector result;
    result.x = v1.x - v2.x;
    result.y = v1.y - v2.y;
    result.z = v1.z - v2.z;
    return result;
}

__device__ Vector multiply(Vector v, double scalar)
{
    Vector result;
    result.x = v.x * scalar;
    result.y = v.y * scalar;
    result.z = v.z * scalar;
    return result;
}

__device__ double magnitude(Vector v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ GridPoint findGridPoint(Vector W)
{
    GridPoint point;
    point.i = (int)(((W.x + Wmax) / (2 * Wmax)) * n);
    point.j = (int)(((W.z + Wmax) / (2 * Wmax)) * n);
    return point;
}

__device__ Vector sampleDirection(curandState_t *state)
{
    double phi = curand_uniform(state) * M_PI;
    double cosTheta = curand_uniform(state)*2 - 1;
    double sinTheta = sqrt(1 - cosTheta * cosTheta);
    double cosPhi = cos(phi);
    double sinPhi = sqrt(1 - cosPhi*cosPhi);
    Vector direction;
    direction.x = sinTheta * cosPhi;
    direction.y = sinTheta * sinPhi;
    direction.z = cosTheta;

    return direction;
}

__global__ void rayTracing(double (*G)[n + 1], Vector C, Vector L, double R, double Wy, int threadsPerBlock, int numBlocks)
{
    int ray = blockIdx.x * blockDim.x + threadIdx.x;
    long long int total = 0;
    curandState_t state;
    curand_init(1234, ray, 0, &state);
    long long int count = 0;
    Vector V, W, I, temp, S, N;
    double t, b;
    GridPoint gridPoint;
    while (count < (Nrays / (threadsPerBlock * numBlocks)))
    {   
       total++;
        V = sampleDirection(&state);
        W = multiply(V, Wy / V.y);
        if (abs(W.x) < Wmax && abs(W.z) < Wmax && (dotProduct(V, C) * dotProduct(V, C)) + R * R - dotProduct(C, C) > 0)
        {
            count++;
            t = dotProduct(V, C) - sqrt((dotProduct(V, C) * dotProduct(V, C)) + R * R - dotProduct(C, C));
            I = multiply(V, t);
            temp = subtract(I, C);
            N = multiply(temp, 1 / magnitude(temp));
            temp = subtract(L, I);
            S = multiply(temp, 1 / magnitude(temp));
            b = dotProduct(S, N) > 0 ? dotProduct(S, N) : 0;
            gridPoint = findGridPoint(W);
            atomicAdd(&G[gridPoint.i][gridPoint.j], b);
        }
    }
    if(ray == 0){ 
        printf("%ld\n",count);
        printf("%ld\n",total);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <threads_per_block> <num_blocks>\n", argv[0]);
        return 1;
    }

    int threadsPerBlock = atoi(argv[1]);
    int numBlocks = atoi(argv[2]);

    double(*G)[n + 1];
    double(*d_G)[n + 1];
    cudaMalloc(&d_G, sizeof(double) * (n + 1) * (n + 1));
    G = (double(*)[n + 1]) malloc(sizeof(double) * (n + 1) * (n + 1));

    Vector C = {0, 12, 0};
    Vector L = {4, 4, -1};
    double R = 6.0;
    double Wy = 2.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    rayTracing<<<numBlocks, threadsPerBlock>>>(d_G, C, L, R, Wy, threadsPerBlock, numBlocks);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    printf("Elapsed Time: %f ms\n", milliseconds);

    cudaMemcpy(G, d_G, sizeof(double) * (n + 1) * (n + 1), cudaMemcpyDeviceToHost);

    FILE *outputFile = fopen("output.txt", "w");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            fprintf(outputFile, "%f ", G[i][j]);
        }
        fprintf(outputFile, "\n");
    }
    fclose(outputFile);

    free(G);
    cudaFree(d_G);
    return 0;
}
