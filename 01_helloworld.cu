__global__ void hello_cuda(){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("[ %d ] Hello World!", idx)
}

int main(){
    hello_cuda<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}