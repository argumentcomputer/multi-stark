#pragma once

namespace {

struct DistributedFinalBuffers {
    int device;
    cudaStream_t stream = nullptr;
    std::unique_ptr<dev_ptr_t<fr_t>> quotient;
    std::array<std::unique_ptr<dev_ptr_t<fr_t>>, 2> staging;
    ~DistributedFinalBuffers() {
        (void)cudaSetDevice(device);
        quotient.reset();
        for (auto& buffer : staging) buffer.reset();
        if (stream) (void)cudaStreamSynchronize(stream);
    }
};

struct DistributedInput {
    QuotientParameters parameters;
    const QuotientColumn* columns;
    size_t column_count, evaluated;
    const QuotientInstruction* nodes;
    size_t node_count;
    const uint32_t* roots;
    size_t root_count;
    const QuotientLookup* lookups;
    size_t lookup_count;
    const uint32_t* args;
    size_t arg_count;
    const fr_t* publics;
    size_t public_count;
    const fr_t* shifts;
    const fr_t* inverse_vanishing;
    size_t n, tile;
};

struct DistributedJob : DistributedLane {
    std::unique_ptr<dev_ptr_t<fr_t>> evaluations, first, partial, scratch;
    std::array<std::unique_ptr<dev_ptr_t<fr_t>>, 2> tiles;
    std::unique_ptr<dev_ptr_t<QuotientColumn>> columns;
    std::unique_ptr<dev_ptr_t<QuotientInstruction>> nodes;
    std::unique_ptr<dev_ptr_t<QuotientLookup>> lookups;
    std::unique_ptr<dev_ptr_t<uint32_t>> roots, args;
    std::unique_ptr<dev_ptr_t<fr_t>> publics;

    DistributedJob(int ordinal, size_t position, int peer, bool force_host)
        : DistributedLane(ordinal, position, peer, force_host) {}
    void clear_sweep() {
        gpu.select();
        evaluations.reset(); first.reset(); scratch.reset();
        for (auto& tile : tiles) tile.reset();
        columns.reset(); nodes.reset(); roots.reset(); lookups.reset(); args.reset(); publics.reset();
        compute.sync();
    }
    ~DistributedJob() {
        gpu.select();
        (void)cudaStreamSynchronize(copy);
        (void)cudaStreamSynchronize(outgoing);
        (void)cudaStreamSynchronize(compute);
        try { clear_sweep(); } catch (...) {}
        partial.reset();
        (void)cudaStreamSynchronize(compute);
    }
    void initialize(const DistributedInput& in) {
        gpu.select();
        const size_t owned = (in.evaluated + 1 - index % 2) / 2;
        const size_t blocks = std::min(size_t(256), (in.tile + 127) / 128);
        evaluations = std::make_unique<dev_ptr_t<fr_t>>(owned * in.n, compute);
        first = std::make_unique<dev_ptr_t<fr_t>>(in.n, compute);
        partial = std::make_unique<dev_ptr_t<fr_t>>(in.n / 2, compute);
        scratch = std::make_unique<dev_ptr_t<fr_t>>(blocks * 128 * in.parameters.slots, compute);
        for (auto& tile : tiles)
            tile = std::make_unique<dev_ptr_t<fr_t>>((in.tile + 1) * in.evaluated, compute);
        columns = std::make_unique<dev_ptr_t<QuotientColumn>>(in.column_count, compute);
        nodes = std::make_unique<dev_ptr_t<QuotientInstruction>>(in.node_count, compute);
        roots = std::make_unique<dev_ptr_t<uint32_t>>(in.root_count, compute);
        lookups = std::make_unique<dev_ptr_t<QuotientLookup>>(in.lookup_count, compute);
        args = std::make_unique<dev_ptr_t<uint32_t>>(in.arg_count, compute);
        publics = std::make_unique<dev_ptr_t<fr_t>>(in.public_count, compute);
        auto& transfer = transfer_lane(gpu);
        auto upload = [&](void* dst, const void* src, size_t bytes) {
            if (bytes) transfer.upload.upload(gpu, compute, dst, src, bytes, stats.transfers);
        };
        upload(columns->data(), in.columns, in.column_count * sizeof(*in.columns));
        upload(nodes->data(), in.nodes, in.node_count * sizeof(*in.nodes));
        upload(roots->data(), in.roots, in.root_count * sizeof(*in.roots));
        upload(lookups->data(), in.lookups, in.lookup_count * sizeof(*in.lookups));
        upload(args->data(), in.args, in.arg_count * sizeof(*in.args));
        upload(publics->data(), in.publics, in.public_count * sizeof(*in.publics));
        distributed_columns(*this, evaluations->data(), in.columns, in.column_count,
                            in.parameters.trace_log, in.shifts[index / 2]);
        timed([&] {
            quotient_ones<<<gpu.sm_count() * 4, 256, 0, compute>>>(first->data(), in.n);
            CUDA_OK(cudaGetLastError());
            transform(gpu, compute, first->data(), in.parameters.trace_log,
                      false, in.shifts[index / 2]);
        });
        ready.record(compute);
        ready.wait();
        transfer.upload.finish_upload(stats.transfers);
        stats.sample(gpu.cid());
    }
};

__global__ __launch_bounds__(128) void quotient_tile_sweep(
    fr_t* output, const fr_t* evaluations, const fr_t* first_selector,
    const QuotientColumn* columns, const QuotientInstruction* nodes, size_t node_count,
    const uint32_t* roots, size_t root_count, const QuotientLookup* lookups, size_t lookup_count,
    const uint32_t* arguments, const fr_t* publics, fr_t* scratch,
    QuotientParameters parameters, fr_t shift, fr_t inverse_vanishing,
    size_t global_start, size_t count, size_t tile_stride
) {
    const size_t n = size_t(1) << parameters.trace_log;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    fr_t point = shift * (parameters.generator ^ uint32_t(global_start + row));
    const fr_t step = parameters.generator ^ uint32_t(stride);
    fr_t* values = scratch + size_t(blockIdx.x) * parameters.slots * blockDim.x + threadIdx.x;
    for (; row < count; row += stride, point *= step) {
        const size_t global = global_start + row;
        const fr_t first = first_selector[global];
        const fr_t last = parameters.generator * first_selector[(global + 1) & (n - 1)];
        const fr_t transition = point - parameters.generator_inverse;
        output[row] = quotient_row(values, evaluations, columns, nodes, node_count,
            roots, root_count, lookups, lookup_count, arguments, publics, parameters,
            tile_stride, row, row + 1, first, last, transition) * inverse_vanishing;
    }
}

__global__ void quotient_interleave_partial(fr_t* output, const fr_t* partial,
                                           size_t global_start, size_t count, size_t coset) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(gridDim.x) * blockDim.x)
        output[(global_start + i) * 2 + coset] = partial[i];
}

void distributed_sweep(std::array<std::unique_ptr<DistributedJob>, 4>& jobs,
                       DistributedJob& job, DistributedRing* ring, const DistributedInput& in) {
    job.gpu.select();
    const size_t start = (job.index % 2) * (in.n / 2);
    const size_t blocks = std::min(size_t(256), (in.tile + 127) / 128);
    for (size_t offset = 0, iteration = 0; offset < in.n / 2; offset += in.tile, ++iteration) {
        const size_t lane = iteration % 2;
        const size_t count = std::min(in.tile, in.n / 2 - offset);
        const size_t global = start + offset;
        distributed_tile(jobs, job, ring, in.n, in.tile, in.evaluated, global, count, lane);
        job.timed([&] {
            quotient_tile_sweep<<<blocks, 128, 0, job.compute>>>(job.partial->data() + offset,
                job.tiles[lane]->data(), job.first->data(), job.columns->data(), job.nodes->data(),
                in.node_count, job.roots->data(), in.root_count, job.lookups->data(), in.lookup_count,
                job.args->data(), job.publics->data(), job.scratch->data(), in.parameters,
                in.shifts[job.index / 2], in.inverse_vanishing[job.index / 2], global, count, in.tile + 1);
            CUDA_OK(cudaGetLastError());
        });
        job.consumed[lane]->record(job.compute);
        job.consumed_recorded[lane] = true;
    }
    if (ring) ring->finish();
    job.done.record(job.compute);
    job.done.wait();
}

} // namespace

extern "C" int multi_stark_kzg_quotient_distributed(
    const int* devices, size_t tile_rows, bool force_host, const QuotientParameters* parameters,
    const QuotientColumn* columns, size_t column_count, size_t evaluation_columns,
    const QuotientInstruction* nodes, size_t node_count, const uint32_t* roots, size_t root_count,
    const QuotientLookup* lookups, size_t lookup_count, const uint32_t* args, size_t arg_count,
    const fr_t* publics, size_t public_count, const fr_t* shifts, const fr_t* inverse_vanishing,
    fr_t* const* outputs
) {
    ProfileRange range("kzg/quotient-distributed");
    try {
        if (!devices || !parameters || !columns || !publics || !shifts || !inverse_vanishing
            || !outputs || !outputs[0] || !outputs[1]) return cudaErrorInvalidValue;
        const auto p = *parameters;
        if (p.trace_log < 10 || p.trace_log > 27 || p.quotient_log != p.trace_log + 1
            || !p.slots || !p.group_size || p.group_size > 8 || p.stage2_start >= column_count
            || public_count < 4 || tile_rows < 128 || tile_rows > (size_t(1) << 20)
            || (tile_rows & (tile_rows - 1)) || (node_count && !nodes) || (root_count && !roots)
            || (lookup_count && !lookups) || (arg_count && !args)) return cudaErrorInvalidValue;
        const size_t n = size_t(1) << p.trace_log;
        if (tile_rows > n / 2 || evaluation_columns > SIZE_MAX / n / sizeof(fr_t)
            || evaluation_columns > SIZE_MAX / (tile_rows + 1) / sizeof(fr_t)) return cudaErrorInvalidValue;
        for (size_t i = 0; i < column_count; ++i) {
            const auto& column = columns[i];
            if (!column.input || !column.count || column.count > n
                || (column.slot != UINT32_MAX && column.slot >= evaluation_columns)
                || (column.resident && column.resident->count != column.count)) return cudaErrorInvalidValue;
        }
        distributed_validate_devices(devices);
        const DistributedInput in{p, columns, column_count, evaluation_columns, nodes, node_count,
            roots, root_count, lookups, lookup_count, args, arg_count, publics, public_count,
            shifts, inverse_vanishing, n, tile_rows};
        std::array<std::unique_ptr<DistributedJob>, 4> jobs;
        std::array<std::unique_ptr<DistributedRing>, 4> tile_rings, merge_rings;
        DistributedFinalBuffers final_buffers{devices[0]};
        // This guard drains every controller's device before shared owners unwind.
        DistributedDrain drain{{devices[0], devices[1], devices[2], devices[3]}};
        distributed_parallel([&](size_t i) {
            jobs[i] = std::make_unique<DistributedJob>(devices[i], i, devices[i ^ 1], force_host);
            jobs[i]->initialize(in);
        });
        for (size_t i = 0; i < 4; ++i) {
            if (force_host) tile_rings[i] = std::make_unique<DistributedRing>(*jobs[i ^ 1], *jobs[i]);
            if (i && (force_host || !distributed_peer(devices[0], devices[i])))
                merge_rings[i] = std::make_unique<DistributedRing>(*jobs[i], *jobs[0]);
        }
        distributed_parallel([&](size_t i) { distributed_sweep(jobs, *jobs[i], tile_rings[i].get(), in); });
        // All peer and staged readers are complete before any source evaluation is freed.
        for (auto& job : jobs) job->clear_sweep();
        auto& primary = *jobs[0];
        primary.gpu.select();
        final_buffers.stream = primary.compute;
        auto& quotient = final_buffers.quotient;
        auto& staging = final_buffers.staging;
        quotient = std::make_unique<dev_ptr_t<fr_t>>(2 * n, primary.compute);
        const size_t chunk = DISTRIBUTED_COPY_BYTES / sizeof(fr_t);
        for (auto& buffer : staging) buffer = std::make_unique<dev_ptr_t<fr_t>>(chunk, primary.compute);
        primary.ready.record(primary.compute);
        primary.ready.wait();
        primary.stats.sample(primary.gpu.cid());
        for (size_t i = 0; i < 4; ++i) {
            auto& source = *jobs[i];
            if (i == 0) {
                primary.timed([&] {
                    quotient_interleave_partial<<<primary.gpu.sm_count() * 4, 256, 0, primary.compute>>>(
                        quotient->data(), source.partial->data(), 0, n / 2, 0);
                    CUDA_OK(cudaGetLastError());
                });
                primary.stats.merge_local_bytes += (n / 2) * sizeof(fr_t);
                primary.stats.transfers.device_copy_bytes += (n / 2) * sizeof(fr_t);
            } else {
                for (size_t offset = 0, iteration = 0; offset < n / 2; offset += chunk, ++iteration) {
                    const size_t lane = iteration % 2, count = std::min(chunk, n / 2 - offset);
                    primary.gpu.select();
                    if (primary.consumed_recorded[lane])
                        CUDA_OK(cudaStreamWaitEvent(primary.copy, primary.consumed[lane]->value, 0));
                    distributed_copy(source, primary, merge_rings[i].get(), staging[lane]->data(),
                                     source.partial->data() + offset, count * sizeof(fr_t), true);
                    primary.copied[lane]->record(primary.copy);
                    primary.gpu.select();
                    CUDA_OK(cudaStreamWaitEvent(primary.compute, primary.copied[lane]->value, 0));
                    primary.timed([&] {
                        quotient_interleave_partial<<<primary.gpu.sm_count() * 4, 256, 0, primary.compute>>>(
                            quotient->data(), staging[lane]->data(), (i % 2) * (n / 2) + offset, count, i / 2);
                        CUDA_OK(cudaGetLastError());
                    });
                    primary.consumed[lane]->record(primary.compute);
                    primary.consumed_recorded[lane] = true;
                }
                if (merge_rings[i]) merge_rings[i]->finish();
            }
            primary.done.record(primary.compute);
            primary.done.wait();
            source.gpu.select();
            source.partial.reset();
            source.compute.sync();
        }
        primary.gpu.select();
        primary.timed([&] { transform(primary.gpu, primary.compute, quotient->data(),
                                      p.quotient_log, true, p.quotient_shift_inverse); });
        auto& transfer = transfer_lane(primary.gpu);
        for (size_t i = 0; i < 2; ++i)
            transfer.download.download(primary.gpu, primary.compute, outputs[i], quotient->data() + i * n,
                                       n * sizeof(fr_t), primary.stats.transfers);
        primary.compute.sync();
        quotient.reset();
        for (auto& buffer : staging) buffer.reset();
        primary.compute.sync();
        for (auto& job : jobs) {
            job->gpu.select();
            for (auto& kernel : job->kernels) kernel->collect(job->stats.transfers);
            job->stats.report(job->gpu.cid(), n / 2);
        }
        return 0;
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}
