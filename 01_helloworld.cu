__global__ void hello_cuda(){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdxx.x;
    printf("[ %d ] hello cuda \n", idx);
}

int main() {
    hello_cuda<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}