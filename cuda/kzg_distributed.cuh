#pragma once

#include <future>

namespace {

constexpr size_t DISTRIBUTED_COPY_BYTES = size_t(16) << 20;

struct DistributedDrain {
    std::array<int, 4> devices;
    ~DistributedDrain() {
        for (int device : devices) {
            (void)cudaSetDevice(device);
            (void)cudaDeviceSynchronize();
        }
    }
};

struct DistributedEvent {
    int device;
    cudaEvent_t value = nullptr;
    explicit DistributedEvent(int owner) : device(owner) {
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaEventCreate(&value));
    }
    ~DistributedEvent() {
        (void)cudaSetDevice(device);
        (void)cudaEventDestroy(value);
    }
    void record(cudaStream_t stream) {
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaEventRecord(value, stream));
    }
    void wait() {
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaEventSynchronize(value));
    }
};

struct DistributedStream {
    int device;
    cudaStream_t value = nullptr;
    explicit DistributedStream(int owner) : device(owner) {
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
    }
    ~DistributedStream() {
        (void)cudaSetDevice(device);
        if (value) {
            (void)cudaStreamSynchronize(value);
            (void)cudaStreamDestroy(value);
        }
    }
    operator cudaStream_t() const { return value; }
};

bool distributed_peer(int accessor, int owner) {
    int supported = 0;
    CUDA_OK(cudaDeviceCanAccessPeer(&supported, accessor, owner));
    return supported != 0;
}

void distributed_pool_access(int accessor, int owner) {
    if (!distributed_peer(accessor, owner)) CUDA_OK(cudaErrorPeerAccessUnsupported);
    CUDA_OK(cudaSetDevice(accessor));
    const auto result = cudaDeviceEnablePeerAccess(owner, 0);
    if (result == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
    else CUDA_OK(result);
    cudaMemPool_t pool = nullptr;
    CUDA_OK(cudaDeviceGetMemPool(&pool, owner));
    cudaMemAccessDesc descriptor{};
    descriptor.location.type = cudaMemLocationTypeDevice;
    descriptor.location.id = accessor;
    descriptor.flags = cudaMemAccessFlagsProtReadWrite;
    cudaMemAccessFlags access{};
    CUDA_OK(cudaMemPoolGetAccess(&access, pool, &descriptor.location));
    if (access != cudaMemAccessFlagsProtReadWrite)
        CUDA_OK(cudaMemPoolSetAccess(pool, &descriptor, 1));
    CUDA_OK(cudaMemPoolGetAccess(&access, pool, &descriptor.location));
    if (access != cudaMemAccessFlagsProtReadWrite) CUDA_OK(cudaErrorPeerAccessUnsupported);
}

void distributed_validate_devices(const int* devices) {
    if (!devices) CUDA_OK(cudaErrorInvalidValue);
    for (size_t i = 0; i < 4; ++i) {
        device_gpu(devices[i]);
        for (size_t j = 0; j < i; ++j)
            if (devices[i] == devices[j]) CUDA_OK(cudaErrorInvalidDevice);
        if (!distributed_peer(devices[i], devices[i ^ 1]))
            CUDA_OK(cudaErrorPeerAccessUnsupported);
    }
}

template<class Operation> void distributed_parallel(Operation operation) {
    std::vector<std::future<void>> workers;
    workers.reserve(4);
    for (size_t i = 0; i < 4; ++i)
        workers.push_back(std::async(std::launch::async, [&, i] { operation(i); }));
    std::exception_ptr failure;
    for (auto& worker : workers) {
        try { worker.get(); }
        catch (...) { if (!failure) failure = std::current_exception(); }
    }
    if (failure) std::rethrow_exception(failure);
}

struct DistributedStats {
    TransferStats transfers;
    size_t coefficient_upload_bytes = 0, coefficient_local_bytes = 0, coefficient_peer_bytes = 0;
    size_t tile_local_bytes = 0, tile_peer_bytes = 0;
    size_t merge_local_bytes = 0, merge_peer_bytes = 0, staged_bytes = 0;
    size_t live_peak_bytes = 0;
    void sample(int device) {
        size_t free = 0, total = 0;
        CUDA_OK(static_cast<cudaError_t>(multi_stark_kzg_memory(device, &free, &total)));
        live_peak_bytes = std::max(live_peak_bytes, total - free);
    }
    void report(int device, size_t elements, const char* operation = "quotient-distributed") {
        transfers.report(device, operation, elements);
        if (transfer_profiling()) std::fprintf(stderr,
            "KZG CUDA distributed device=%d coefficient_upload_bytes=%zu coefficient_local_bytes=%zu coefficient_peer_bytes=%zu "
            "tile_local_bytes=%zu tile_peer_bytes=%zu merge_local_bytes=%zu merge_peer_bytes=%zu "
            "staged_bytes=%zu live_peak_bytes=%zu\n", device, coefficient_upload_bytes,
            coefficient_local_bytes, coefficient_peer_bytes, tile_local_bytes, tile_peer_bytes, merge_local_bytes,
            merge_peer_bytes, staged_bytes, live_peak_bytes);
    }
};

struct DistributedLane {
    const gpu_t& gpu;
    stream_t& compute;
    size_t index;
    int partner;
    bool force_host;
    DistributedStream copy, outgoing;
    DistributedEvent ready, done;
    std::array<std::unique_ptr<DistributedEvent>, 2> copied, consumed;
    std::array<bool, 2> consumed_recorded{};
    std::vector<std::unique_ptr<TimedKernel>> kernels;
    DistributedStats stats;
    DistributedLane(int ordinal, size_t position, int peer, bool stage_host)
        : gpu(device_gpu(ordinal)), compute(gpu[0]), index(position), partner(peer), force_host(stage_host),
          copy(ordinal), outgoing(ordinal),
          ready(ordinal), done(ordinal) {
        for (size_t i = 0; i < 2; ++i) {
            copied[i] = std::make_unique<DistributedEvent>(ordinal);
            consumed[i] = std::make_unique<DistributedEvent>(ordinal);
        }
    }
    ~DistributedLane() {
        gpu.select();
        (void)cudaStreamSynchronize(copy);
        (void)cudaStreamSynchronize(outgoing);
        (void)cudaStreamSynchronize(compute);
    }
    template<class Operation> void timed(Operation operation) {
        gpu.select();
        auto timer = std::make_unique<TimedKernel>();
        timer->start(compute);
        operation();
        timer->end(compute);
        kernels.push_back(std::move(timer));
    }
};

void distributed_columns(DistributedLane& job, fr_t* evaluations,
                         const QuotientColumn* columns, size_t column_count,
                         uint32_t trace_log, fr_t shift) {
    const size_t n = size_t(1) << trace_log;
    auto& transfer = transfer_lane(job.gpu);
    for (size_t i = 0; i < column_count; ++i) {
        const auto& column = columns[i];
        if (column.slot == UINT32_MAX || column.slot % 2 != job.index % 2) continue;
        fr_t* destination = evaluations + size_t(column.slot / 2) * n;
        const size_t bytes = column.count * sizeof(fr_t);
        if (column.resident && column.resident->device == job.gpu.cid()) {
            CUDA_OK(cudaMemcpyAsync(destination, column.resident->data, bytes,
                                   cudaMemcpyDeviceToDevice, job.compute));
            job.stats.coefficient_local_bytes += bytes;
            job.stats.transfers.device_copy_bytes += bytes;
        } else if (!job.force_host && column.resident && column.resident->device == job.partner) {
            // The immutable source stays borrowed until the acquired pair drains.
            CUDA_OK(cudaMemcpyPeerAsync(destination, job.gpu.cid(), column.resident->data,
                                       job.partner, bytes, job.compute));
            job.stats.coefficient_peer_bytes += bytes;
            job.stats.transfers.device_copy_bytes += bytes;
        } else {
            transfer.upload.upload(job.gpu, job.compute, destination, column.input, bytes, job.stats.transfers);
            job.stats.coefficient_upload_bytes += bytes;
        }
        if (column.count < n)
            CUDA_OK(cudaMemsetAsync(destination + column.count, 0,
                                   (n - column.count) * sizeof(fr_t), job.compute));
        job.timed([&] { transform(job.gpu, job.compute, destination, trace_log,
                              false, shift); });
    }
}

class DistributedRing {
    struct Slot {
        void* host = nullptr;
        DistributedEvent produced_start, produced, consumed_start, consumed;
        bool pending = false;
        Slot(int source, int destination)
            : produced_start(source), produced(source), consumed_start(destination), consumed(destination) {
            CUDA_OK(cudaHostAlloc(&host, DISTRIBUTED_COPY_BYTES, cudaHostAllocPortable));
        }
        ~Slot() { if (host) (void)cudaFreeHost(host); }
    };
    DistributedLane& source;
    DistributedLane& destination;
    std::array<std::unique_ptr<Slot>, 2> slots;
    size_t next = 0;
    void finish(Slot& slot) {
        if (!slot.pending) return;
        slot.consumed.wait();
        float elapsed = 0;
        source.gpu.select();
        CUDA_OK(cudaEventElapsedTime(&elapsed, slot.produced_start.value, slot.produced.value));
        destination.stats.transfers.download_ms += elapsed;
        destination.gpu.select();
        CUDA_OK(cudaEventElapsedTime(&elapsed, slot.consumed_start.value, slot.consumed.value));
        destination.stats.transfers.upload_ms += elapsed;
        slot.pending = false;
    }
  public:
    DistributedRing(DistributedLane& from, DistributedLane& to) : source(from), destination(to) {
        for (auto& slot : slots) slot = std::make_unique<Slot>(source.gpu.cid(), destination.gpu.cid());
    }
    void copy(void* dst, const void* src, size_t bytes) {
        for (size_t offset = 0; offset < bytes; offset += DISTRIBUTED_COPY_BYTES) {
            auto& slot = *slots[next++ % slots.size()];
            finish(slot);
            const size_t count = std::min(DISTRIBUTED_COPY_BYTES, bytes - offset);
            slot.produced_start.record(source.outgoing);
            CUDA_OK(cudaMemcpyAsync(slot.host, static_cast<const unsigned char*>(src) + offset,
                                   count, cudaMemcpyDeviceToHost, source.outgoing));
            slot.produced.record(source.outgoing);
            slot.produced.wait();
            slot.consumed_start.record(destination.copy);
            CUDA_OK(cudaMemcpyAsync(static_cast<unsigned char*>(dst) + offset, slot.host,
                                   count, cudaMemcpyHostToDevice, destination.copy));
            slot.consumed.record(destination.copy);
            slot.pending = true;
        }
        destination.stats.transfers.download_bytes += bytes;
        destination.stats.transfers.upload_bytes += bytes;
        destination.stats.staged_bytes += bytes;
    }
    void finish() { for (auto& slot : slots) finish(*slot); }
};

void distributed_copy(DistributedLane& source, DistributedLane& destination,
                      DistributedRing* ring, void* dst, const void* src, size_t bytes, bool merge) {
    if (!bytes) return;
    destination.gpu.select();
    if (source.index == destination.index) {
        CUDA_OK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, destination.copy));
        (merge ? destination.stats.merge_local_bytes : destination.stats.tile_local_bytes) += bytes;
        destination.stats.transfers.device_copy_bytes += bytes;
    } else if (ring) {
        ring->copy(dst, src, bytes);
    } else {
        CUDA_OK(cudaMemcpyPeerAsync(dst, destination.gpu.cid(), src, source.gpu.cid(), bytes, destination.copy));
        (merge ? destination.stats.merge_peer_bytes : destination.stats.tile_peer_bytes) += bytes;
        destination.stats.transfers.device_copy_bytes += bytes;
    }
}

template<class Job> void distributed_tile(
    std::array<std::unique_ptr<Job>, 4>& jobs, Job& job, DistributedRing* ring,
    size_t n, size_t tile, size_t evaluated, size_t global, size_t count, size_t lane
) {
    job.gpu.select();
    if (job.consumed_recorded[lane])
        CUDA_OK(cudaStreamWaitEvent(job.copy, job.consumed[lane]->value, 0));
    for (size_t column = 0; column < evaluated; ++column) {
        auto& owner = *jobs[(job.index / 2) * 2 + column % 2];
        const fr_t* source = owner.evaluations->data() + (column / 2) * n;
        fr_t* destination = job.tiles[lane]->data() + column * (tile + 1);
        const size_t contiguous = std::min(count + 1, n - global);
        distributed_copy(owner, job, owner.index == job.index ? nullptr : ring,
                         destination, source + global, contiguous * sizeof(fr_t), false);
        if (contiguous != count + 1)
            distributed_copy(owner, job, owner.index == job.index ? nullptr : ring,
                             destination + contiguous, source, sizeof(fr_t), false);
    }
    job.copied[lane]->record(job.copy);
    job.gpu.select();
    CUDA_OK(cudaStreamWaitEvent(job.compute, job.copied[lane]->value, 0));
}

} // namespace

extern "C" int multi_stark_kzg_peer_access(int accessor, int owner, int* supported) {
    if (!supported) return cudaErrorInvalidValue;
    return cudaDeviceCanAccessPeer(supported, accessor, owner);
}

extern "C" int multi_stark_kzg_distributed_peers(const int* devices) {
    try {
        distributed_validate_devices(devices);
        for (size_t accessor = 0; accessor < 4; ++accessor)
            for (size_t owner = 0; owner < 4; ++owner)
                if (accessor != owner && distributed_peer(devices[accessor], devices[owner]))
                    distributed_pool_access(devices[accessor], devices[owner]);
        return 0;
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}
