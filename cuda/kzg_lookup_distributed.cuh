#pragma once

namespace {

struct DistributedLookupInput {
    LookupParameters parameters;
    const QuotientColumn* columns;
    size_t column_count, evaluated;
    const QuotientInstruction* nodes;
    size_t node_count;
    const QuotientLookup* lookups;
    size_t lookup_count;
    const uint32_t* args;
    size_t arg_count;
    size_t n, rows, groups, count, tile;
};

struct DistributedLookupJob : DistributedLane {
    std::unique_ptr<dev_ptr_t<fr_t>> evaluations, scratch;
    std::array<std::unique_ptr<dev_ptr_t<fr_t>>, 2> tiles;
    std::unique_ptr<dev_ptr_t<QuotientColumn>> columns;
    std::unique_ptr<dev_ptr_t<QuotientInstruction>> nodes;
    std::unique_ptr<dev_ptr_t<QuotientLookup>> lookups;
    std::unique_ptr<dev_ptr_t<uint32_t>> args;
    std::unique_ptr<dev_ptr_t<fr_t>> numerators, denominators, forward, reverse, inverse;
    std::unique_ptr<dev_ptr_t<fr_t>> offsets, output;

    DistributedLookupJob(int ordinal, size_t position, int peer, bool force_host)
        : DistributedLane(ordinal, position, peer, force_host) {}
    void clear_sweep() {
        gpu.select();
        evaluations.reset(); scratch.reset();
        for (auto& tile : tiles) tile.reset();
        columns.reset(); nodes.reset(); lookups.reset(); args.reset();
        compute.sync();
    }
    void clear_scan() {
        gpu.select();
        numerators.reset(); forward.reset(); reverse.reset(); inverse.reset(); offsets.reset();
        compute.sync();
    }
    ~DistributedLookupJob() {
        gpu.select();
        (void)cudaStreamSynchronize(copy);
        (void)cudaStreamSynchronize(outgoing);
        (void)cudaStreamSynchronize(compute);
        try { clear_sweep(); clear_scan(); } catch (...) {}
        denominators.reset(); output.reset();
        (void)cudaStreamSynchronize(compute);
    }
    void initialize(const DistributedLookupInput& in) {
        gpu.select();
        const size_t owned = (in.evaluated + 1 - index % 2) / 2;
        const size_t blocks = std::min(size_t(256), (in.tile + 127) / 128);
        evaluations = std::make_unique<dev_ptr_t<fr_t>>(owned * in.n, compute);
        numerators = std::make_unique<dev_ptr_t<fr_t>>(in.count, compute);
        denominators = std::make_unique<dev_ptr_t<fr_t>>(in.count, compute);
        scratch = std::make_unique<dev_ptr_t<fr_t>>(blocks * 128 * in.parameters.slots, compute);
        for (auto& tile : tiles)
            tile = std::make_unique<dev_ptr_t<fr_t>>((in.tile + 1) * in.evaluated, compute);
        columns = std::make_unique<dev_ptr_t<QuotientColumn>>(in.column_count, compute);
        nodes = std::make_unique<dev_ptr_t<QuotientInstruction>>(in.node_count, compute);
        lookups = std::make_unique<dev_ptr_t<QuotientLookup>>(in.lookup_count, compute);
        args = std::make_unique<dev_ptr_t<uint32_t>>(in.arg_count, compute);
        auto& transfer = transfer_lane(gpu);
        auto upload = [&](void* dst, const void* src, size_t bytes) {
            if (bytes) transfer.upload.upload(gpu, compute, dst, src, bytes, stats.transfers);
        };
        upload(columns->data(), in.columns, in.column_count * sizeof(*in.columns));
        upload(nodes->data(), in.nodes, in.node_count * sizeof(*in.nodes));
        upload(lookups->data(), in.lookups, in.lookup_count * sizeof(*in.lookups));
        upload(args->data(), in.args, in.arg_count * sizeof(*in.args));
        distributed_columns(*this, evaluations->data(), in.columns, in.column_count,
                            in.parameters.trace_log, fr_t::one());
        ready.record(compute);
        ready.wait();
        transfer.upload.finish_upload(stats.transfers);
        stats.sample(gpu.cid());
    }
    void scan(const DistributedLookupInput& in, fr_t* total) {
        gpu.select();
        forward = std::make_unique<dev_ptr_t<fr_t>>(in.count, compute);
        reverse = std::make_unique<dev_ptr_t<fr_t>>(in.count, compute);
        inverse = std::make_unique<dev_ptr_t<fr_t>>(1, compute);
        timed([&] {
            prefix_op<Multiply<fr_t>>(forward->data(), denominators->data(), in.count, compute);
            prefix_op<Multiply<fr_t>>(ReverseOutput{reverse->data(), in.count},
                                     ReverseInput{denominators->data(), in.count}, in.count, compute);
            lookup_inverse_product<<<1, 32, 0, compute>>>(inverse->data(), forward->data() + in.count - 1);
            CUDA_OK(cudaGetLastError());
            lookup_contributions<<<gpu.sm_count() * 4, 256, 0, compute>>>(
                numerators->data(), forward->data(), reverse->data(), inverse->data(), in.count);
            CUDA_OK(cudaGetLastError());
            prefix_op<Add<fr_t>>(numerators->data(), in.count, compute);
        });
        transfer_lane(gpu).download.download(gpu, compute, total,
            numerators->data() + in.count - 1, sizeof(fr_t), stats.transfers);
        compute.sync();
        stats.sample(gpu.cid());
    }
};

__global__ __launch_bounds__(128) void lookup_tile_rationals(
    fr_t* numerators, fr_t* denominators, const fr_t* evaluations,
    const QuotientColumn* columns, const QuotientInstruction* nodes, size_t node_count,
    const QuotientLookup* lookups, size_t lookup_count, const uint32_t* arguments,
    fr_t* scratch, LookupParameters parameters,
    size_t global_start, size_t count, size_t tile_stride
) {
    const size_t n = size_t(1) << parameters.trace_log;
    const size_t groups = (lookup_count + parameters.group_size - 1) / parameters.group_size;
    fr_t zero;
    zero.zero();
    const fr_t one = fr_t::one();
    fr_t* values = scratch + size_t(blockIdx.x) * parameters.slots * blockDim.x + threadIdx.x;
    for (size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += size_t(gridDim.x) * blockDim.x) {
        const size_t global = global_start + row;
        const fr_t first = fr_t::csel(one, zero, global == 0);
        const fr_t last = fr_t::csel(one, zero, global + 1 == n);
        const fr_t transition = fr_t::csel(zero, one, global + 1 == n);
        lookup_rational_row(numerators + row * groups, denominators + row * groups,
            values, evaluations, columns, nodes, node_count, lookups, lookup_count,
            arguments, parameters, tile_stride, row, row + 1, first, last, transition);
    }
}

void distributed_lookup_sweep(std::array<std::unique_ptr<DistributedLookupJob>, 4>& jobs,
                              DistributedLookupJob& job, DistributedRing* ring,
                              const DistributedLookupInput& in) {
    const size_t start = job.index * in.rows;
    const size_t blocks = std::min(size_t(256), (in.tile + 127) / 128);
    for (size_t offset = 0, iteration = 0; offset < in.rows; offset += in.tile, ++iteration) {
        const size_t lane = iteration % 2, count = std::min(in.tile, in.rows - offset);
        const size_t global = start + offset;
        distributed_tile(jobs, job, ring, in.n, in.tile, in.evaluated, global, count, lane);
        job.timed([&] {
            lookup_tile_rationals<<<blocks, 128, 0, job.compute>>>(
                job.numerators->data() + offset * in.groups,
                job.denominators->data() + offset * in.groups, job.tiles[lane]->data(),
                job.columns->data(), job.nodes->data(), in.node_count, job.lookups->data(),
                in.lookup_count, job.args->data(), job.scratch->data(), in.parameters,
                global, count, in.tile + 1);
            CUDA_OK(cudaGetLastError());
        });
        job.consumed[lane]->record(job.compute);
        job.consumed_recorded[lane] = true;
    }
    if (ring) ring->finish();
    job.done.record(job.compute);
    job.done.wait();
}

__global__ void lookup_partition_offsets(fr_t* totals) {
    if (threadIdx.x == 0) {
        fr_t sum;
        sum.zero();
        for (size_t i = 0; i < 4; ++i) {
            const fr_t value = totals[i];
            totals[i] = sum;
            sum += value;
        }
        totals[4] = sum;
    }
}

__global__ void lookup_partition_columns(fr_t* columns, const fr_t* inclusive,
                                         size_t rows, size_t groups, fr_t offset) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < rows * groups; i += size_t(gridDim.x) * blockDim.x) {
        const size_t group = i / rows, row = i % rows;
        const size_t index = row * groups + group;
        fr_t value = offset;
        if (index) value += inclusive[index - 1];
        columns[i] = value;
    }
}

} // namespace

extern "C" int multi_stark_kzg_lookup_distributed(
    const int* devices, size_t tile_rows, bool force_host, const LookupParameters* parameters,
    const QuotientColumn* columns, size_t column_count, size_t evaluation_columns,
    const QuotientInstruction* nodes, size_t node_count,
    const QuotientLookup* lookups, size_t lookup_count, const uint32_t* args, size_t arg_count,
    fr_t* const* outputs, fr_t* total
) {
    ProfileRange range("kzg/lookup-distributed");
    try {
        if (!devices || !parameters || !columns || !lookups || !lookup_count || !outputs || !total)
            return cudaErrorInvalidValue;
        const auto p = *parameters;
        if (p.trace_log < 10 || p.trace_log > 27 || !p.slots || !p.group_size || p.group_size > 8
            || tile_rows < 128 || tile_rows > (size_t(1) << 20) || (tile_rows & (tile_rows - 1))
            || (node_count && !nodes) || (arg_count && !args)) return cudaErrorInvalidValue;
        const size_t n = size_t(1) << p.trace_log, rows = n / 4;
        const size_t groups = (lookup_count + p.group_size - 1) / p.group_size;
        if (groups > SIZE_MAX / n / sizeof(fr_t) || tile_rows > rows
            || evaluation_columns > SIZE_MAX / n / sizeof(fr_t)
            || evaluation_columns > SIZE_MAX / (tile_rows + 1) / sizeof(fr_t)) return cudaErrorInvalidValue;
        for (size_t group = 0; group < groups; ++group)
            if (!outputs[group]) return cudaErrorInvalidValue;
        for (size_t i = 0; i < column_count; ++i) {
            const auto& column = columns[i];
            if (!column.input || !column.count || column.count > n
                || (column.slot != UINT32_MAX && column.slot >= evaluation_columns)
                || (column.resident && column.resident->count != column.count)) return cudaErrorInvalidValue;
        }
        distributed_validate_devices(devices);
        const DistributedLookupInput in{p, columns, column_count, evaluation_columns, nodes,
            node_count, lookups, lookup_count, args, arg_count, n, rows, groups, rows * groups, tile_rows};
        std::array<std::unique_ptr<DistributedLookupJob>, 4> jobs;
        std::array<std::unique_ptr<DistributedRing>, 4> tile_rings, gather_rings;
        std::array<fr_t, 5> offsets;
        // Owners outlive the guard so failures drain every peer reader before freeing sources.
        DistributedDrain drain{{devices[0], devices[1], devices[2], devices[3]}};
        distributed_parallel([&](size_t i) {
            jobs[i] = std::make_unique<DistributedLookupJob>(devices[i], i, devices[i ^ 1], force_host);
            jobs[i]->initialize(in);
        });
        for (size_t i = 0; i < 4; ++i)
            if (force_host) tile_rings[i] = std::make_unique<DistributedRing>(*jobs[i ^ 1], *jobs[i]);
        distributed_parallel([&](size_t i) { distributed_lookup_sweep(jobs, *jobs[i], tile_rings[i].get(), in); });
        for (auto& job : jobs) job->clear_sweep();
        distributed_parallel([&](size_t i) { jobs[i]->scan(in, &offsets[i]); });

        auto& primary = *jobs[0];
        primary.gpu.select();
        primary.offsets = std::make_unique<dev_ptr_t<fr_t>>(5, primary.compute);
        auto& transfer = transfer_lane(primary.gpu);
        transfer.upload.upload(primary.gpu, primary.compute, primary.offsets->data(), offsets.data(),
                               4 * sizeof(fr_t), primary.stats.transfers);
        primary.timed([&] {
            lookup_partition_offsets<<<1, 32, 0, primary.compute>>>(primary.offsets->data());
            CUDA_OK(cudaGetLastError());
        });
        transfer.download.download(primary.gpu, primary.compute, offsets.data(), primary.offsets->data(),
                                   5 * sizeof(fr_t), primary.stats.transfers);
        primary.compute.sync();
        transfer.upload.finish_upload(primary.stats.transfers);
        *total = offsets[4];
        distributed_parallel([&](size_t i) {
            auto& job = *jobs[i];
            job.timed([&] {
                lookup_partition_columns<<<job.gpu.sm_count() * 4, 256, 0, job.compute>>>(
                    job.denominators->data(), job.numerators->data(), rows, groups, offsets[i]);
                CUDA_OK(cudaGetLastError());
            });
            job.compute.sync();
            job.clear_scan();
        });

        for (size_t batch = 0; batch < groups; batch += 4) {
            const size_t active = std::min(size_t(4), groups - batch);
            distributed_parallel([&](size_t i) {
                if (i >= active) return;
                auto& job = *jobs[i];
                job.gpu.select();
                job.output = std::make_unique<dev_ptr_t<fr_t>>(n, job.compute);
                job.ready.record(job.compute);
                job.ready.wait();
                job.stats.sample(job.gpu.cid());
            });
            for (size_t round = 0; round < 4; ++round) {
                for (size_t i = 0; i < active; ++i) {
                    const size_t source = (i + round) % 4;
                    gather_rings[i].reset();
                    if (source != i && (force_host || !distributed_peer(devices[i], devices[source])))
                        gather_rings[i] = std::make_unique<DistributedRing>(*jobs[source], *jobs[i]);
                }
                // A permutation gives each source stream exactly one host controller per round.
                distributed_parallel([&](size_t i) {
                    if (i >= active) return;
                    const size_t source = (i + round) % 4;
                    auto& job = *jobs[i];
                    distributed_copy(*jobs[source], job, gather_rings[i].get(),
                        job.output->data() + source * rows,
                        jobs[source]->denominators->data() + (batch + i) * rows,
                        rows * sizeof(fr_t), true);
                    if (gather_rings[i]) gather_rings[i]->finish();
                    job.done.record(job.copy);
                    job.done.wait();
                });
            }
            distributed_parallel([&](size_t i) {
                if (i >= active) return;
                auto& job = *jobs[i];
                job.timed([&] {
                    transform(job.gpu, job.compute, job.output->data(), p.trace_log, true, fr_t::one());
                });
                transfer_lane(job.gpu).download.download(job.gpu, job.compute,
                    outputs[batch + i], job.output->data(), n * sizeof(fr_t), job.stats.transfers);
                job.compute.sync();
                job.output.reset();
                job.compute.sync();
            });
        }
        for (auto& job : jobs) {
            job->gpu.select();
            job->denominators.reset();
            job->compute.sync();
            for (auto& kernel : job->kernels) kernel->collect(job->stats.transfers);
            job->stats.report(job->gpu.cid(), rows, "lookup-distributed");
        }
        return 0;
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}
