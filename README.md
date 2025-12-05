## 模型优化

- Graph: 在不改变模型最终数学逻辑的前提下，通过修改计算流程的“蓝图”（计算图）来提升效率，例如将多个小步骤合并成一个大步骤（算子融合，QKVParallelLinear）。
- Op:专注于将每一个独立的计算步骤（算子，如卷积、加法）本身，用针对特定硬件（如CPU或GPU）最高效的代码来实现（conv2d ➡ matmul）。
- Runtime: 在模型真正执行推理时，通过智能地调度计算任务、管理内存等方式，最大限度地压榨硬件性能，减少等待和浪费（PageAttention, Continuous Batching）。

## AI推理加速（算力，I/O，访存，通信）

- 计算效率：取决于硬件算力，以及计算的持续而不被打断
- 访存效率：取决于访存延迟和带宽
- 计算与访存相重叠
- 计算与通信相重叠

## 模型大小评估指标

1. 计算量 (Computational Cost)
   定义：计算量反映了模型对硬件计算单元的需求。计算量的单位通常是 OPs (Operations) 或 FLOPs (Floating Point Operations)，即浮点运算次数。在深度学习中，最常用的数据格式为 float32，因此 float32 类型下的计算量常被写作 FLOPs。模型的整体计算量等于模型中每个算子的计算量之和。

例子：对两个 shape 为 (N, C, H, W) 的 float32 tensor 进行 add 操作，其计算量为 N×C×H×W FLOPs。

2. 参数量 (Number of Parameters)
   定义：模型中所有需要学习的参数的总和，它直接反映了模型占用的磁盘空间大小。

说明：对于 CNN（卷积神经网络）而言，参数主要由 Conv (卷积) 层和 FC (全连接) 层的 Weight (权重) 构成。其他算子（如 BatchNorm、激活函数等）虽然也可能有参数，但与前者相比通常占比较小。

3. 访存量 (Memory Access)
   定义：指模型计算时所需访问内存/显存的字节大小，它反映了模型对内存/显存带宽的需求。访存量单位为 Bytes，表示模型计算到底需要读取/取多少 Bytes 的数据。

   例子：对两个 shape 为 (N, C, H, W) 的 float32 tensor 进行 add 操作，访存量为 (2+1)×N×C×H×W×sizeof(float32) bytes。其中“2”代表读取两个输入张量，“1”代表写入一个输出张量。
4. (峰值)内存占用 (Peak Memory Usage)

- 定义：指模型跑起来的时候（训练或推理）所占用的内存/显存大小。峰值内存占用，特指在运行过程中的内存/显存占用的峰值。注意：内存占用 ≠ 访存量。
- 峰值内存占用的特征描述：

  - 动态性 (Dynamic)：峰值内存是一个动态指标，它在模型单次迭代（iteration）的执行过程中是不断变化的。如上图所示，内存在前向传播（forward）阶段通常会持续增长，在反向传播（backward）阶段会因为部分中间变量被释放而有所波动和下降。
  - 综合性 (Comprehensive)：它不仅包括模型本身的参数（权重），还包括训练或推理过程中产生的中间激活值（feature maps）、计算出的梯度（在训练时），以及优化器（Optimizer）自身的状态（如动量信息）等。
  - 训练远高于推理：通常情况下，模型训练时的峰值内存占用远大于推理时。这是因为训练需要保存前向传播过程中的所有中间激活值，以便在反向传播时计算梯度，同时还要存储梯度本身和优化器的状态。而推理过程通常只需要保留前一层的输出供后一层使用，可以做到“阅后即焚”。
  - 非简单叠加：峰值内存并不是所有张量（参数、激活值、梯度）大小的简单相加。现代深度学习框架有高效的内存管理机制，会复用内存空间。峰值内存反映的是在某一特定时刻，同时存在于内存/显存中的所有张量的总大小的最大值。

## 模型中常见的量化算子

模型量化并非对所有算子都一视同仁，它主要针对那些计算密集且能量消耗大的算子，以及那些容易用整数运算来模拟的算子。

以下是常见的、可以被有效量化的算子类型，我将它们分为几类以便于理解。

#### 1. 核心计算密集型算子 (Compute-Intensive Operators)

这类算子是量化的**最主要目标**，因为它们占据了模型大部分的计算时间。将它们从浮点运算转换为整数运算，能带来最大的性能提升。

* **卷积类 (Convolutions)**
  * `Conv2D` (二维卷积)
  * `DepthwiseConv2D` (深度可分离卷积)
  * `Conv3D` (三维卷积)
  * `ConvTranspose2D` (转置卷积 / 反卷积)
* **全连接/线性类 (Fully-Connected / Linear)**
  * `MatMul` (矩阵乘法)
  * `FullyConnected` / `Linear` (全连接层)
  * `BatchMatMul` (批量矩阵乘法)

这些算子的核心都是大量的乘法-累加（Multiply-Accumulate, MAC）操作，非常适合在硬件上用高效的整数指令集来执行。

#### 2. 逐元素操作算子 (Element-wise Operations)

这类算子对张量的每个元素进行独立操作，通常也易于量化。

* **基础算术**: `Add` (加法), `Mul` (乘法), `Sub` (减法)
  * *注意*: 对两个量化张量进行这些操作时，通常要求它们的量化参数（scale和zero-point）兼容或需要进行额外的重量化（Requantization）步骤。
* **比较**: `Maximum`, `Minimum`

#### 3. 激活函数 (Activation Functions)

激活函数的量化分为两种情况：

* **易于量化的激活函数**:
  * `ReLU`: `max(0, x)`，在整数域非常容易实现。
  * `ReLU6`: `min(max(0, x), 6)`，同样是简单的比较和裁剪操作。
  * `LeakyReLU` 及其他 PReLU、ReLU 的变体。
* **较难量化的激活函数**:
  * `Sigmoid`
  * `Tanh`
  * `Swish` / `SiLU`
  * `GeLU`
  * 这些函数是非线性的，并且形状复杂。在量化时，通常使用**查找表 (Look-up Table, LUT)** 或者分段多项式逼近的方法来模拟它们的行为，会引入一定的精度误差。

#### 4. 池化和塑形算子 (Pooling and Shaping Operators)

* **池化类 (Pooling)**:
  * `MaxPool` (最大池化)
  * `AveragePool` (平均池化)
  * `GlobalAveragePool` (全局平均池化)
* **数据塑形/重排类 (Shaping / Reordering)**:
  * `Concat` (拼接)：通常要求待拼接的张量有相同的量化参数。
  * `Reshape` (重塑形状)
  * `Transpose` (转置)
  * `Squeeze` / `Unsqueeze` (维度压缩/扩展)
  * `Pad` (填充)
  * `Slice` / `Split` (切分)

#### 5. 归一化算子 (Normalization Operators)

* **批归一化 (Batch Normalization)**: `BatchNorm` 本身在推理时**不会被直接量化**。它是一个线性变换，因此在部署前，它的参数（均值、方差、gamma、beta）会被数学上**“折叠”或“融合”（Fuse）**到前一层的卷积或全连接层的权重和偏置中。您上一张图中就展示了这个过程。
* **其他归一化**:
  * `LayerNorm`, `InstanceNorm`: 这些比 `BatchNorm` 更难处理，因为它们的归一化统计量是动态计算的，量化起来更复杂，有时为了精度会被保留为浮点计算。

### 总结


| 类别             | 常见算子                       | 量化友好度                  |
| :----------------- | :------------------------------- | :---------------------------- |
| **计算密集型**   | `Conv2D`, `MatMul`, `Linear`   | **高 (主要目标)**           |
| **激活函数**     | `ReLU`, `ReLU6`                | **高**                      |
|                  | `Sigmoid`, `Tanh`, `Swish`     | 中 (需要查找表或近似)       |
| **池化与塑形**   | `MaxPool`, `AvgPool`, `Concat` | 高                          |
| **逐元素操作**   | `Add`, `Mul`                   | 高 (需注意量化参数)         |
| **归一化**       | `BatchNorm`                    | **高 (通过融合实现)**       |
|                  | `LayerNorm`                    | 低 (通常保持浮点)           |
| **其他复杂函数** | `Softmax`, `exp`, `log`, `pow` | 低 (通常保持浮点或用查找表) |

总而言之，一个模型中绝大多数的算子都可以被量化，但核心收益来自于对**卷积和全连接层**的量化。对于复杂的非线性函数，则需要在性能和精度之间做出权衡。不同的量化框架（如 TensorFlow Lite, TensorRT, PyTorch Quantization）和目标硬件支持的算子列表也会略有不同。

## **CUDA 程序中的修饰符**

这些修饰符用于指定函数在何种处理器（CPU 或 GPU）上执行和被调用。

1. **`__global__`**

   * **说明**：这是一个 CUDA 核函数 (Kernel) 的前缀。
   * **调用与执行**：该函数由 **CPU (主机端)** 调用启动，在 **GPU (设备端)** 上执行。
2. **`__host__`**

   * **说明**：表示一个主机端函数。
   * **调用与执行**：该函数由 **CPU (主机端)** 调用，并在 **CPU (主机端)** 上执行。
   * **注意**：正常的 C++ 函数如果没有加任何修饰符，默认就是 `__host__` 函数。
3. **`__device__`**

   * **说明**：表示一个设备端函数。
   * **调用与执行**：该函数只能被 **GPU (设备端)** 调用（通常由 `__global__` 函数或其他 `__device__` 函数调用），并在 **GPU (设备端)** 上执行。

---

### **组合使用**

* `__host__` 和 `__device__` 可以一起使用，表示该函数既可以被 CPU 调用并在 CPU 上运行，也可以被 GPU 调用并在 GPU 上运行。编译器会为这份代码生成两个版本。

  **示例**:

  ```cpp
  __host__ __device__ int run_on_cpu_or_gpu() {
      return 1;
  }
  ```

---

### **内联修饰符**

这些修饰符向编译器提供关于函数内联（inline）的指令。

4. **`__noinline__`**

   * **说明**：强制编译器**不要**将该函数进行内联展开。
5. **`__forceinline__`**

   * **说明**：强制编译器**必须**将该函数进行内联展开。
   * **注意**：`__forceinline__` 通常与 `__device__` 一起使用，以优化设备端代码的性能。

您说得对，我确实遗漏了图片中关于**60%利用率**这个重要的图示信息。这个阈值可以作为判断Kernel类型的直观标准。

这是修正和补充后的版本：

## **CUDA Kernel 的三大瓶颈类型**

1. **计算密集型 (Compute Bound)**

   * **判定**:
     * **公式**:
       $$
       \frac{\text{计算量}}{\text{访存量}} > \frac{\text{GPU峰值算力}}{\text{GPU峰值带宽}}

       $$
     * **图示**: 计算单元利用率 (Comp) **> 60%**，而访存利用率 (Mem) 较低。
   * **瓶颈**: 性能受限于GPU的算术运算速度。
   * **例子**: 大规模矩阵乘法 (gemm)、卷积 (conv)。
2. **访存密集型 (Memory Bound)**

   * **判定**:
     * **公式**:
       $$
       \frac{\text{计算量}}{\text{访存量}} < \frac{\text{GPU峰值算力}}{\text{GPU峰值带宽}}

       $$
     * **图示**: 访存利用率 (Mem) **> 60%**，而计算单元利用率 (Comp) 较低。
   * **瓶颈**: 性能受限于从显存读写数据的速度（带宽）。
   * **例子**: ReLU、数组拼接 (concat)。
3. **延迟密集型 (Latency Bound)**

   * **判定**:
     * **图示**: 计算 (Comp) 和访存 (Mem) 的利用率**均远低于 60%**。
   * **瓶颈**: 性能受限于指令依赖、同步或数据规模过小导致的流水线停顿。
   * **例子**: 未经优化的、或处理小尺寸数据 (small shape) 的 Kernel。

## **CUDA 计算流程**

```
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sum(float *x)
{
    // 泛指当前block在所有block范围内的id
    int block_id = blockIdx.x;
    // 泛指当前线程在所有block范围内的全局id
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 泛指当前线程在其block内的id
    int local_tid = threadIdx.x;
    printf("current block=%d, thread id in current block =%d, global thread id=%d\n", block_id, local_tid, global_tid);
    x[global_tid] += 1;
}

int main(){
    int N = 32;
    int nbytes = N * sizeof(float);
    float *dx, *hx;
    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);//思考为什么要用二级指针
    /* allocate CPU mem */
    hx = (float*) malloc(nbytes);
    /* init host data */
    printf("hx original: \n");
    for (int i = 0; i < N; i++) {
        hx[i] = i;
        printf("%g\n", hx[i]);
    }
    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    /* launch GPU kernel */
    sum<<<1, N>>>(dx);
    /* copy data from GPU */
    cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);
    printf("hx current: \n");
    for (int i = 0; i < N; i++) {
        printf("%g\n", hx[i]);
    }
    cudaFree(dx);
    free(hx);
    return 0;
}
```

![](assets/20251008_160006_image.png)

![](assets/20251008_160056_image.png)

![](assets/20251008_160129_image.png)

![](assets/20251008_160201_image.png)

## CUDA 程序的线程层次结构（按从小到大）

- **Thread（线程）**: 由 CUDA 运行时分配，每个线程有自己的寄存器与程序计数器，通过 `threadIdx` 等标识；硬件上对应到一个执行通道（lane）。
- **Warp（束/扭）**: GPU 的基本调度单位。NVIDIA 架构中每个 warp 固定为 32 个线程（历史或跨厂商文档可能写“最多 32”/32 或 64，NVIDIA 现行就是 32）。同一 warp 内线程在硬件上锁步执行。
- **Block（线程块）**: 用户定义的线程集合，通过 `blockIdx` 标识。一个 block 由多个 warp 组成，能共享片上共享内存与同步（如 `__syncthreads()`）。
- **Grid（网格）**: 由一个或多个 block 构成，一次 kernel 启动对应一个 grid。

### 关于 warp（补充要点）

- **执行模型**: CUDA 采用 SIMT（Single Instruction, Multiple Threads）。
- **调度**: warp 由硬件的 warp scheduler 与 dispatch port 发射到执行单元。
- **开销**: CUDA 的 warp/线程调度被设计为“接近零开销”；创建线程的额外成本约为一个时钟级别。
- **逻辑 vs 硬件视图**: 逻辑上是一个 thread block；硬件上被切分成每组 32 线程的多个 warp；执行时由控制逻辑把 warp 交给 SM 的执行单元。

## CUDA 错误检测宏

```
// 错误宏定义
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// 错误宏用法
gpuErrchk(cudaMalloc((void**)&A_d, ARRAY_SIZE*sizeof(float)));
gpuErrchk(cudaMemcpy(C_d, C_h, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice));
gpuErrchk(cudaMemcpy(A_h, A_d, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
```

## **CUDA main函数编写流程**

* host端申请内存并初始化数据
* device端申请内存
* host端数据拷贝到device端
* (计时开始)
* 启动CUDA kernel
* (计时结束)
* device端把结果拷贝回host端
* 检查device端计算结果和host端计算结果
* 释放host和device端内存

## 向量化的向量加法

- 加速vectorAdd，向量化load & store
- 可以用来测量GPU显存带宽

## CPU架构

![alt text](image-2.png)

## CPU执行流程

![alt text](image.png)

CPU详细的执行步骤如下：

* **取指令 (Fetch)：** 流程始于CPU内部的**指令指针寄存器 (IP)**，它储存了下一条待执行指令在内存中的地址。CPU根据此地址，从内存 (DRAM) 的**代码段**中获取指令。
* **解码 (Decode)：** 获取到的指令首先被送到**指令寄存器 (IR)**。接着，**指令译码器 (ID)** 会对该指令进行分析和解码，以确定需要执行什么操作以及如何执行。这个过程由**操作控制器 (OC)** 进行协调。
* **执行 (Execute)：**
  * 指令解码后，如果需要数据，CPU会从内存 (DRAM) 的**数据段**中提取所需数据。
  * 这些数据会被加载到CPU内部的**储存单元 (SRAM)**。
  * **控制单元**发出信号，将指令和数据发送给**运算单元**进行计算。
* **更新指针与循环：** 在一个指令周期执行的同时，指令指针寄存器 (IP) 会更新并指向下一条指令的地址，以便下一个周期的取指操作，从而形成一个不断循环的流水线过程。

## GPU架构

![alt text](image-1.png)

## GPU架构简介/CUDA执行模型

![alt text](image-3.png)
从大到小来理解这个映射：

#### 1. 网格 (Grid) ➔ 设备 (Device)

* **软件层面 (Grid - 网格)**：当你启动一个CUDA核函数（Kernel），你就定义了一个`Grid`。`Grid`是执行同一个核函数的所有线程的集合，是软件上并行任务的最高层级。你可以把它想象成**整个计算任务**。
* **硬件层面 (Device - 设备)**：`Device`指的就是整个GPU芯片。一个`Grid`中的所有计算任务都会被分配到这块GPU上执行。

**关系：一个完整的计算任务（Grid）运行在一整个GPU（Device）上。**

#### 2. 线程块 (Thread Block) ➔ 流多处理器 (SM - Streaming Multiprocessor)

* **软件层面 (Thread Block - 线程块)**：一个`Grid`由多个`Thread Block`组成。`Block`是一组可以相互协作的线程集合（例如，通过共享内存 `Shared Memory` 进行通信）。程序员可以定义一个Block里有多少个线程。你可以把它想象成一个**紧密协作的“施工队”**。
* **硬件层面 (SM - 流多处理器)**：`SM`是GPU的核心硬件单元，你可以把它看作一块GPU上的一个“迷你处理器”或**“车间”**。它包含了CUDA核心、调度器、寄存器文件、共享内存等执行资源。
* **映射关系**：程序启动后，操作系统和CUDA驱动会将一个或多个`Thread Block`（施工队）**调度到**一个可用的`SM`（车间）上执行。**一个Block内的所有线程保证会在同一个SM上执行完毕**，但一个SM可以同时执行多个Block。

**关系：一个“线程块”（Block）会被完整地调度到一个“流多处理器”（SM）上执行。这是性能调优的关键，因为同一Block内的线程可以利用SM上的高速共享内存进行快速数据交换。**

#### 3. 线程 (Thread) 与 Warp ➔ CUDA核心 (CUDA Core)

* **软件层面 (Thread - 线程)**：`Thread`是执行任务的最小单位，也就是程序员编写代码的视角。你写的每一行核函数代码，都是从一个`Thread`的角度去执行的。你可以把它想象成一个**“工人”**。
* **硬件层面 (Warp & CUDA Core)**：这是最关键也最需要理解的一层。
  * **硬件并不以单个线程为单位进行调度**。为了提高效率，SM中的**Warp Scheduler (Warp调度器)**会将32个线程捆绑成一个**`Warp`**。
  * `Warp`是GPU硬件**调度和执行的最小单位**。你可以把它看作一个由32名工人组成的**“班组”**。这个班组里的所有工人（线程）步调完全一致。
  * 在某个时钟周期，Warp调度器会发射一条指令，这个Warp中的**32个线程**会**同时**在SM内的多个**CUDA Core**上执行**同一条指令**，但各自处理的是不同的数据。这个执行模式就是图上提到的**SIMT (Single Instruction, Multiple Thread)**。

**关系：程序员写的单个`Thread`只是一个逻辑概念。在硬件执行时，32个`Thread`会被打包成一个`Warp`，以`Warp`为单位进行调度。一个`Warp`中的所有线程在同一时刻执行相同的指令，跑在不同的CUDA Core上。**


| 软件抽象 (Software Abstraction) | 硬件实体 (Hardware Reality) | 关系描述                                                             |
| :-------------------------------- | :---------------------------- | :--------------------------------------------------------------------- |
| **Grid (网格)**                 | **Device (GPU设备)**        | 整个计算任务在整个GPU上运行。                                        |
| **Thread Block (线程块)**       | **SM (流多处理器)**         | 一个线程块被分配到一个SM上执行，块内线程可高效协作。                 |
| **Thread (线程)**               | **CUDA Core (CUDA核心)**    | 程序员定义的最小逻辑单位，最终在CUDA核心上执行计算。                 |
| *(无直接软件对应)*              | **Warp (线程束)**           | **硬件调度的最小单位**，由32个线程组成，是连接软件和硬件的关键桥梁。 |

![alt text](image-4.png)
**通用公式是：**
PeakFLOPS = F_clk * N_SM * Ops_per_SM_per_cycle

其中：

- F_clk: GPU的核心频率（这里是1.41 GHz）
- N_SM: GPU拥有的流多处理器（SM）总数（这里是108个）
- Ops_per_SM_per_cycle: 每个SM每个周期能完成的运算次数（这个数据就来自上方的表格）

## GPU存储结构

![](assets/20251015_163202_image.png)

### 一、六种存储媒介特点总结

| 存储类型 | 位置 | 缓存 | 访问 | 作用域 | 生命周期 | 延迟 | 特点 |
|---------|------|------|------|-------|---------|------|------|
| **寄存器** | 片上 | - | R/W | 线程 | 线程 | 1 周期 | 最快，容量极小 |
| **本地内存** | 片外 | Yes | R/W | 线程 | 线程 | 400+ 周期 | 寄存器溢出，慢 |
| **共享内存** | 片上 | - | R/W | 块 | 块 | 1-2 周期 | 块内协作，需同步 |
| **全局内存** | 片外 | Yes | R/W | 全局+主机 | 主机分配 | 400-800 周期 | 容量大，速度慢 |
| **常量内存** | 片外 | Yes | R | 全局+主机 | 主机分配 | 缓存命中快 | 只读，广播优化 |
| **纹理内存** | 片外 | Yes | R | 全局+主机 | 主机分配 | 缓存命中快 | 只读，空间局部性 |

---

### 二、核心概念

#### **片上 vs 片外**
- **片上**（寄存器、共享内存）：集成在 GPU 芯片内，延迟 1-2 周期，带宽 ~TB/s 级，容量 KB 级
- **片外**（全局内存等）：外部 DRAM，延迟 400-800 周期（**慢 400 倍**），容量 GB 级

#### **作用域含义**
- **线程级**：每个线程独立，无需同步，无法通信
- **块级**：块内线程共享，需 `__syncthreads()` 同步，不同块隔离
- **全局级**：所有线程+主机可访问，需原子操作或分离 kernel 同步

---

### 三、三大核心存储关系

#### **1. 全局内存（Global Memory）**
- **角色**：主数据仓库，CPU-GPU 数据交换桥梁
- **特点**：容量大（GB 级），速度慢，所有线程可访问
- **用途**：输入/输出数据，大规模数组，跨块通信

#### **2. 共享内存（Shared Memory）**
- **角色**：块内高速缓存
- **特点**：容量小（48-164KB/块），速度快（片上），块内共享
- **用途**：数据重用，线程协作，减少全局内存访问

#### **3. 寄存器（Registers）**
- **角色**：线程私有存储
- **特点**：容量极小，速度最快，线程独占
- **用途**：临时变量，循环计数器，频繁访问的局部数据

---

### 四、典型数据流与优化

**数据流动**：
```
CPU → 全局内存（慢，片外） 
      ↓ 合并读取
   共享内存（快，片上，块级）
      ↓ 分配
   寄存器（最快，片上，线程级）
      ↓ 计算
   共享内存 → 全局内存 → CPU
```

**优化原则**：
1. 减少全局内存访问（用共享内存缓存）
2. 合并访问（连续线程访问连续地址）
3. 避免 bank 冲突（共享内存）
4. 控制寄存器使用（避免溢出到本地内存）

**同步需求**：
- 线程级（寄存器）：❌ 无需同步
- 块级（共享内存）：✅ `__syncthreads()`
- 全局级（全局内存）：✅ 原子操作 / 分离 kernel

**核心目标**：数据尽量在片上、作用域尽量小、访问模式合并、减少同步开销。

## 深度学习算子类型总结

- Gemm类：matmul, batch_matmul
- Sliding window类：conv2d, conv3d, maxpool ➡️ img2col ➡️ Gemm
- **Reduce类**：softmax
- **Elementwise类**: gelu, copy_if (以元素为单位的操作，矩阵加法等)
- **Fused类**：MatmulAndBiasAndRelu, fusedMaskScaleAndAdd
- Scan类：prefixsum, cumsum
- Sort类：mergesort
- 坐标变换类：concat, transpose
