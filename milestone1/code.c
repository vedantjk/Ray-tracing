#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define n 512     // Grid dimensions
#define Wmax 2.0 // Window bounds

typedef struct
{
    double x, y, z;
} Vector;

typedef struct
{
    int i, j;
} GridPoint;

double dotProduct(Vector v1, Vector v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

Vector subtract(Vector v1, Vector v2)
{
    Vector result;
    result.x = v1.x - v2.x;
    result.y = v1.y - v2.y;
    result.z = v1.z - v2.z;
    return result;
}

Vector add(Vector v1, Vector v2)
{
    Vector result;
    result.x = v1.x + v2.x;
    result.y = v1.y + v2.y;
    result.z = v1.z + v2.z;
    return result;
}

double magnitude(Vector v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vector multiply(Vector v, double scalar)
{
    Vector result;
    result.x = v.x * scalar;
    result.y = v.y * scalar;
    result.z = v.z * scalar;
    return result;
}

GridPoint findGridPoint(Vector W)
{
    GridPoint point;
    point.i = (int)(((W.x + Wmax) / (2 * Wmax)) * n);
    point.j = (int)(((W.z + Wmax) / (2 * Wmax)) * n);
    return point;
}

Vector sampleDirection()
{
    double phi = ((double)rand() / RAND_MAX) * 2 * M_PI;
    double cosTheta = ((double)rand() / RAND_MAX) * 2 - 1;
    double sinTheta = sqrt(1 - cosTheta * cosTheta);

    Vector direction;
    direction.x = sinTheta * cos(phi);
    direction.y = sinTheta * sin(phi);
    direction.z = cosTheta;

    return direction;
}

int main()
{
    double G[n + 1][n + 1] = {{0.0}}; // Initialize grid
    int Nrays = n * n * n;
    Vector C = {0, 12, 0};
    Vector L = {4, 4, -1};
    double R = 6.0;
    double Wy = 2.0;
    Vector V, W, I, temp, S, N;
    double t, b;
    GridPoint gridPoint;
    for (int ray = 0; ray < Nrays; ray++)
    {
        Vector V, W;

        while (1)
        {
            V = sampleDirection();
            W = multiply(V, Wy / V.y);
            if (abs(W.x) < Wmax && abs(W.z) < Wmax && (dotProduct(V, C) * dotProduct(V, C)) + R * R - dotProduct(C, C) > 0)
                break;
        }
        t = dotProduct(V, C) - sqrt((dotProduct(V, C) * dotProduct(V, C)) + R * R - dotProduct(C, C));
        I = multiply(V, t);
        temp = subtract(I, C);
        N = multiply(temp, 1 / magnitude(temp));
        temp = subtract(L, I);
        S = multiply(temp, 1 / magnitude(temp));
        b = fmax(0, dotProduct(S, N));
        gridPoint = findGridPoint(W);
        G[gridPoint.i][gridPoint.j] += b;
    }
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

    return 0;
}
