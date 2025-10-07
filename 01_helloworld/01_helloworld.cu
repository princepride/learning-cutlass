#include <stdio.h>

__global__ void hello_cuda(){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("[%d] Hello Cuda", idx);
}

int main(){
    hello_cuda<<<4, 2>>>();
    cudaDeviceSynchronize();
    return 0;
}

