To run the multi GPU codes I used the following command - `mpiexec --oversubscribe -np 2 ./a.out 1000000000 100 80 512
`. Change the values as required. The -np should be number of GPUs to run. There is a Make file in each folder except for the multi GPU folders
where I have given a `build.sh` file.
