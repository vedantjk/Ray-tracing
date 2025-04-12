#ifdef __cplusplus
extern "C"
{
#endif

    void computeGPU(float *G, int Nrays, int n, int threadsPerBlock, int numBlocks);

#ifdef __cplusplus
}
#endif
