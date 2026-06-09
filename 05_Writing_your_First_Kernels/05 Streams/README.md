# CUDA Streams Examples

## Intuition
You can think of streams as "river streams" where the direction of operations flows only forward in time (like a timeline). For example, copy some data over (time step 1), then do some computation (time step 2), then copy some data back (time step 3). This is the basic idea behind streams. 

We can have multiple streams at once in CUDA, and each stream can have its own timeline. This allows us to overlap operations and make better use of the GPU.

When training a massive language model, it would be silly to spend a ton of time loading all the tokens in and out of the GPU. Streams allow us to move data around while also doing computation at all times. Streams introduce a software abstraction called "prefetching", which is a way to move data around before it is needed. This is a way to hide the latency of moving data around. 

This project demonstrates the usage of CUDA streams for concurrent execution and better GPU utilization. It contains two examples:


## Code Snippets
- default **stream** = **stream** 0 = null **stream**
```cpp
// This kernel launch uses the null stream (0)
myKernel<<<gridSize, blockSize>>>(args);

// This is equivalent to
myKernel<<<gridSize, blockSize, 0, 0>>>(args);
```

Remember this part from the Kernels section?
- The execution configuration (of a global function call) is specified by inserting an expression of the form `<<<gridDim, blockDim, Ns, S>>>`, where:

  - Dg (dim3) specifies the dimension and size of the grid.
  - Db (dim3) specifies the dimension and size of each block
  - Ns (size_t) specifies the number of bytes in shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory. (typically omitted)
  - S (cudaStream_t) specifies the associated stream, is an optional parameter which defaults to 0.

- stream 1 and stream 2 are created with different priorities. this means they are executed in a certain order at runtime. this essentially gives us more control over the concurrent execution of our kernels.

```cpp
    // Create streams with different priorities
    int leastPriority, greatestPriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));
```

## Examples

1. `stream_basics.cu`: Illustrates basic stream usage with asynchronous memory transfers and kernel launches.
2. `stream_advanced.cu`: Demonstrates more advanced concepts like stream priorities, callbacks, and inter-stream dependencies.

## Compilation

To compile the examples, use the following commands:

```bash
nvcc -o 01 01_stream_basics.cu
nvcc -o 02 02_stream_advanced.cu
```

这段代码带你进入了 CUDA 进阶性能优化的核心领域：流水线并发（Concurrency via CUDA Streams）。
在默认情况下，你写的所有 CUDA 代码都是在“默认流（Default Stream/Null Stream）”里跑的，里面的任务必须像老太太过马路一样——一个接一个地排队串行。
而这段代码通过显式创建两个 Stream（流），打破了这种限制。下面我们用大白话和图解，把 CUDA Stream 的前因后果一次性讲透。
🌊 什么是 CUDA Stream（流）？
一句话定义： CUDA Stream 是指在 GPU 上按顺序执行的一组异步任务队列。
你可以把 GPU 想象成一个巨大的超级工厂，里面有“运输队”（负责把数据在 CPU 和 GPU 之间搬来搬去）和“生产车间”（负责跑 Kernel 计算）。
同一个 Stream 内： 所有任务严格排队。第一步没搬完，第二步的计算绝对不会开始。
不同的 Stream 之间： 彼此独立。这意味着，Stream 1 的运输队在搬数据时，Stream 2 的生产车间可以同时在算另一批数据！ 这种让“数据传输”和“计算”在时间上重叠（Overlap）的技术，是大厂优化大模型训练/推理吞吐量的必杀技。
🛠️ 结合代码：逐行拆解 Stream 的高级用法
这段代码展示了如何利用两个流（stream1 和 stream2）进行异步操作。我们来硬核拆解它的核心部分：
### 1. 声明与创建流
```
C++
cudaStream_t stream1, stream2; // 声明两个流的句柄

// 在 GPU 内部真正创建这两条任务流水线
CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));
```
### 2. 异步数据传输 (cudaMemcpyAsync)
普通的 cudaMemcpy 是阻塞的，CPU 必须死等数据搬完才能执行下一行。而代码里用了高级的 cudaMemcpyAsync（异步传输）：
```
C++
// 告诉 stream1：去把 h_A 发送到 d_A。下达完命令后，CPU 立刻转头执行下一行！
CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1));

// 告诉 stream2：去把 h_B 发送到 d_B。
CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2));
```
性能优势： 这两行搬运命令几乎是同时下达给 GPU 的。由于它们在不同的流里，GPU 内部如果有多个复制引擎（Copy Engines），d_A 和 d_B 的传输在硬件层面甚至可以产生并发，大大节省了等待时间。
### 3. 同步特定的流 (cudaStreamSynchronize)
```
C++
// CPU 走到这里停下，死等 stream2 里的数据（也就是 d_B）全部搬完。搬完才准往下走。
CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));
```
❓ 为什么要在这里单独卡死同步 stream2？
因为下一行我们要调用计算核函数 vectorAdd<<<...>>>(d_A, d_B, d_C, ...)。这个计算需要同时用到 d_A 和 d_B。
我们把这个计算任务指派给了 stream1。因为 d_A 本来就在 stream1 里排队，所以计算任务一定会等 d_A 搬完才执行；但是 d_B 是隔壁 stream2 搬的，stream1 无法感知 stream2 的进度！
为了防止 stream1 已经开工算加法了，d_B 还没搬完（导致算出来一堆垃圾脏数据），代码在这里非常严谨地让 CPU 踩了个刹车，确保 d_B 已经安全抵达显存。
### 4. 指派核函数到指定的流
```
C++
// 执行配置的第 4 个参数（stream1）代表：把这个计算任务塞进 stream1 的队列里
vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, numElements);
```
### 5. 传回结果与整体收工
```
C++
// 告诉 stream1：等上面的 vectorAdd 算完后，把结果 d_C 异步搬回 CPU 端的 h_C
CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1));

// 终极大检查：CPU 死等 stream1 把计算和最后的传输彻底干完
CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
```
### 6. 打扫战场（销毁流）
流是占用 GPU 硬件资源的（比如硬件连接上下文），用完必须像释放内存一样销毁掉：
```
C++
CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
```
💡 为什么这段代码的 Stream 用法其实“不够完美”？（大厂面试高频扣分点）
如果你把这段代码直接拿去大厂参加 AI 芯片开发或 vLLM 核心库团队的面试，面试官可能会微微一笑，指出这段代码有一个隐藏的性能瑕疵（Bug）：
致命细节：它用了普通的 malloc 申请主机内存！
在 CUDA 中，要想让 cudaMemcpyAsync（异步复制）真正实现“CPU 和 GPU 一边算一边搬”的真·异步效果，主机端（CPU）的内存必须是页锁定内存（Pinned Memory / Page-locked Memory）。
如果像代码里这样用普通的 malloc，系统内存是可以被操作系统随意换页（Page fault）的。这会导致底层的驱动破功，迫使 cudaMemcpyAsync 自动降级退化成同步阻塞模式，让你的多流并发大打折扣。
🚀 正确的大厂工程级写法：
应该把 CPU 端的 malloc 替换为 cudaHostAlloc 或者 cudaMallocHost：
```
C++
// 错误/低效的写法：
// h_A = (float *)malloc(size);

// 正确/完美开启异步流性能的写法：
CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_A, size));
CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_B, size));
CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_C, size));

// 结尾释放时使用：
cudaFreeHost(h_A);
```
总结
Stream 是 GPU 的多线程流水线，用来并发现象。
不同的 Stream 可以让 传输（Copy） 和 计算（Kernel） 同时发生，榨干 GPU 的每一滴血。
必须配合 Pinned Memory（cudaMallocHost），才能真正释放 Stream 异步流的全部威力。

## Docs
- https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf

## Pinned Memory
- "we're gonna need this for later, so don't play with it" is a good way to think about it.
- pinned memory is memory that is locked in place and cannot be moved around by the OS. This is useful for when you want to move data to the GPU and do some computation on it. If the OS moves the data around, the GPU will be looking for the data in the wrong place and you will get a segfault.
```cpp
// Allocate pinned memory
float* h_data;
cudaMallocHost((void**)&h_data, size);
```

## Events
- Measuring kernel execution time: Events are placed before and after kernel launches to measure execution time accurately.

- Synchronizing between streams: Events can be used to create dependencies between different streams, ensuring one operation starts only after another has completed.

- Overlapping computation and data transfer: Events can mark the completion of a data transfer, signaling that computation can begin on that data.


```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

## Callbacks
-  By using callbacks, you can set up a pipeline where the completion of one operation on the GPU triggers the start of another operation on the CPU, which might then queue more work for the GPU. (as seen in the nvidia concurrency docs above)

```cpp
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("GPU operation completed\n");
    // Trigger next batch of work
}

kernel<<<grid, block, 0, stream>>>(args);
cudaStreamAddCallback(stream, MyCallback, nullptr, 0);
```
