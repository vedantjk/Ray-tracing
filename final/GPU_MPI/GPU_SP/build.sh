#!/bin/bash

mpic++ -c gpuMPI.cpp -o main.o
nvcc -c gpuMPI.cu -o main_g.o -arch=sm_70 
mpic++ main.o main_g.o -lcudart