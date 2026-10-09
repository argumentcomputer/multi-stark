#include <ff/bls12-381.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <ntt/ntt.cuh>
#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#define TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
#include <msm/pippenger.cuh>
#include <polynomial/evaluate.cuh>

#include <exception>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "kzg_transfer.cuh"

using point_t = jacobian_t<fp_t>;
using bucket_t = xyzz_t<fp_t>;
using affine_t = bucket_t::affine_t;
using msm_impl = msm_t<bucket_t, point_t, affine_t, fr_t>;

static_assert(sizeof(fr_t) == 32);
static_assert(sizeof(affine_t) == 96);
static_assert(sizeof(point_t) == 144);

static int failure(int code, const char* message) {
    // Complete queued host transfers before Rust releases its limb buffers.
    (void)cudaDeviceSynchronize();
    std::fprintf(stderr, "KZG CUDA: %s\n", message);
    return code;
}

extern "C" int multi_stark_kzg_devices(int* count) {
    try {
        *count = static_cast<int>(ngpus());
        return 0;
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}

extern "C" int multi_stark_kzg_device_ordinal(int index, int* ordinal) {
    try {
        if (!ordinal || index < 0 || static_cast<size_t>(index) >= ngpus())
            return cudaErrorInvalidDevice;
        *ordinal = all_gpus()[index]->cid();
        return 0;
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}

static const gpu_t& device_gpu(int ordinal) {
    const gpu_t* found = nullptr;
    for (const auto* gpu : all_gpus())
        if (gpu->cid() == ordinal) found = gpu;
    if (!found) CUDA_OK(cudaErrorInvalidDevice);
    return select_gpu(found->id());
}

extern "C" int multi_stark_kzg_memory(int device, size_t* free, size_t* total) {
    try {
        device_gpu(device);
        return cudaMemGetInfo(free, total);
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}

namespace {
struct MsmContext {
    size_t count;
    dev_ptr_t<affine_t> points;
    msm_impl msm;
    MsmContext(const gpu_t& gpu, size_t count)
        : count(count), points(count, gpu),
          msm(nullptr, count, sizeof(affine_t), gpu.id()) {}
};

int msm_result(RustError status) {
    if (status.code)
        failure(status.code, status.message ? status.message : cudaGetErrorString(
            static_cast<cudaError_t>(status.code)));
    std::free(status.message);
    return status.code;
}
} // namespace

// Coordinates and scalars are explicit little-endian Montgomery limbs;
// affine infinity is (0, 0), projective infinity has Z = 0.
extern "C" int multi_stark_kzg_msm_create(int device, void** out,
                                          const affine_t* points, size_t count) {
    try {
        if (!out || !points || count == 0 || count > (size_t(1) << 28))
            return cudaErrorInvalidValue;
        // Resident chunk points avoid the streaming path's cross-stream use of
        // an allocation before its cudaMallocAsync stream has completed.
        const auto& gpu = device_gpu(device);
        auto context = std::make_unique<MsmContext>(gpu, count);
        TransferStats stats;
        auto& transfer = transfer_lane(gpu);
        transfer.upload.upload(gpu, gpu, context->points, points, count * sizeof(affine_t), stats);
        gpu.sync();
        transfer.upload.finish_upload(stats);
        stats.report(device, "msm-points", count);
        *out = context.release();
        return 0;
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}

extern "C" int multi_stark_kzg_msm_invoke(int device, void* context, point_t* out,
                                          const fr_t* scalars, size_t count) {
    try {
        if (!context || !out || !scalars || count == 0)
            return cudaErrorInvalidValue;
        const auto& gpu = device_gpu(device);
        auto& msm = *static_cast<MsmContext*>(context);
        if (count > msm.count) return cudaErrorInvalidValue;
        dev_ptr_t<fr_t> data(count, gpu);
        TransferStats stats;
        auto& transfer = transfer_lane(gpu);
        transfer.upload.upload(gpu, gpu, data, scalars, count * sizeof(fr_t), stats);
        gpu.sync();
        transfer.upload.finish_upload(stats);
        const int code = msm_result(msm.msm.invoke(*out, msm.points, count, data, true));
        stats.report(device, "msm-scalars", count);
        return code;
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}

extern "C" int multi_stark_kzg_msm_destroy(int device, void* context) {
    try {
        const auto& gpu = device_gpu(device);
        delete static_cast<MsmContext*>(context);
        gpu.sync();
        return 0;
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}

__global__ void scale_coset(fr_t* data, size_t count, fr_t shift) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    fr_t power = shift ^ static_cast<uint32_t>(i);
    const fr_t step = shift ^ static_cast<uint32_t>(stride);
    for (; i < count; i += stride, power *= step)
        data[i] *= power;
}

namespace {

struct ResidentPolynomial {
    int device;
    size_t count;
    dev_ptr_t<fr_t> data;
    ResidentPolynomial(const gpu_t& gpu, size_t count, const stream_t& stream)
        : device(gpu.cid()), count(count), data(count, stream) {}
};

struct FftJob {
    const fr_t* input;
    const ResidentPolynomial* resident_input;
    fr_t* output;
    size_t input_count;
    ResidentPolynomial** resident_output;
};

void transform(const gpu_t& gpu, stream_t& stream, fr_t* data,
               uint32_t lg, bool inverse, const fr_t& shift) {
    if (lg == 0) return;
    const size_t count = size_t(1) << lg;
    if (!shift.is_one() && !inverse) {
        scale_coset<<<gpu.sm_count() * 4, 256, 0, stream>>>(data, count, shift);
        CUDA_OK(cudaGetLastError());
    }
    NTT::Base_dev_ptr(stream, data, lg, NTT::InputOutputOrder::NN,
                     inverse ? NTT::Direction::inverse : NTT::Direction::forward,
                     NTT::Type::standard);
    if (!shift.is_one() && inverse) {
        scale_coset<<<gpu.sm_count() * 4, 256, 0, stream>>>(data, count, shift);
        CUDA_OK(cudaGetLastError());
    }
}

} // namespace

extern "C" int multi_stark_kzg_polynomial_upload(int device, const fr_t* input,
                                                  size_t count, void** output) {
    try {
        if (!input || !output || !count) return cudaErrorInvalidValue;
        const auto& gpu = device_gpu(device);
        auto resident = std::make_unique<ResidentPolynomial>(gpu, count, gpu);
        auto& transfer = transfer_lane(gpu);
        TransferStats stats;
        transfer.upload.upload(gpu, gpu, resident->data, input, count * sizeof(fr_t), stats);
        gpu.sync();
        transfer.upload.finish_upload(stats);
        stats.report(device, "retain", count);
        *output = resident.release();
        return 0;
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}

extern "C" int multi_stark_kzg_polynomial_destroy(int device, void* pointer) {
    try {
        const auto& gpu = device_gpu(device);
        delete static_cast<ResidentPolynomial*>(pointer);
        gpu.sync();
        return 0;
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}

__global__ void scale_values(fr_t* output, const fr_t* input, size_t count, fr_t scale) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(gridDim.x) * blockDim.x)
        output[i] = input[i] * scale;
}

extern "C" int multi_stark_kzg_msm_invoke_resident(int device, void* context,
                                                  point_t* out, const void* polynomial,
                                                  size_t offset, size_t count,
                                                  const fr_t* normalization) {
    try {
        if (!context || !out || !polynomial || !count) return cudaErrorInvalidValue;
        const auto& gpu = device_gpu(device);
        const auto& coefficients = *static_cast<const ResidentPolynomial*>(polynomial);
        auto& msm = *static_cast<MsmContext*>(context);
        if (count > msm.count || coefficients.device != device ||
            offset > coefficients.count || count > coefficients.count - offset)
            return cudaErrorInvalidValue;
        dev_ptr_t<fr_t> normalized(normalization ? count : 0, gpu);
        fr_t* input = const_cast<fr_t*>(&coefficients.data[offset]);
        if (normalization) {
            scale_values<<<gpu.sm_count() * 4, 256, 0, gpu>>>(normalized, input, count, *normalization);
            CUDA_OK(cudaGetLastError());
            gpu.sync();
            input = normalized;
        }
        dev_ptr_t<fr_t> scalars(input, count);
        return msm_result(msm.msm.invoke(*out, msm.points, count, scalars, true));
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}

extern "C" int multi_stark_kzg_fft_batch(int device, const FftJob* jobs,
                                          size_t job_count, uint32_t lg, bool inverse,
                                          const fr_t* shift, size_t slots) {
    try {
        if (!jobs || !shift || lg > 31 || slots < 1 || slots > 2)
            return cudaErrorInvalidValue;
        const size_t count = size_t(1) << lg;
        for (size_t i = 0; i < job_count; ++i) {
            const auto& job = jobs[i];
            if (!job.output || job.input_count > count ||
                (job.input_count && !job.input && !job.resident_input) ||
                (job.resident_input && (job.resident_input->device != device ||
                                       job.resident_input->count != job.input_count)))
                return cudaErrorInvalidValue;
        }
        const auto& gpu = device_gpu(device);
        struct Slot {
            std::unique_ptr<ResidentPolynomial> buffer;
            const FftJob* job = nullptr;
            TimedKernel kernel;
            TransferStats stats;
        };
        {
            std::array<Slot, 2> queue;
            const auto drain = [&](size_t index) {
                auto& slot = queue[index];
                if (!slot.job) return;
                auto& transfers = transfer_lane(gpu, index);
                transfers.download.download(gpu, gpu[index], slot.job->output,
                    slot.buffer->data, count * sizeof(fr_t), slot.stats);
                transfers.upload.finish_upload(slot.stats);
                slot.kernel.collect(slot.stats);
                slot.stats.report(device, inverse ? "ifft" : "fft", count);
                if (slot.job->resident_output)
                    *slot.job->resident_output = slot.buffer.release();
                slot.job = nullptr;
            };
            for (size_t i = 0; i < job_count; ++i) {
                const size_t index = i % slots;
                drain(index);
                auto& slot = queue[index];
                auto& stream = gpu[index];
                auto& transfers = transfer_lane(gpu, index);
                const auto& job = jobs[i];
                if (!slot.buffer)
                    slot.buffer = std::make_unique<ResidentPolynomial>(gpu, count, stream);
                slot.stats = TransferStats{};
                if (job.resident_input) {
                    CUDA_OK(cudaMemcpyAsync(slot.buffer->data, job.resident_input->data,
                        job.input_count * sizeof(fr_t), cudaMemcpyDeviceToDevice, stream));
                    slot.stats.device_copy_bytes += job.input_count * sizeof(fr_t);
                } else {
                    transfers.upload.upload(gpu, stream, slot.buffer->data, job.input,
                                            job.input_count * sizeof(fr_t), slot.stats);
                }
                if (job.input_count < count)
                    CUDA_OK(cudaMemsetAsync(&slot.buffer->data[job.input_count], 0,
                                           (count - job.input_count) * sizeof(fr_t), stream));
                slot.kernel.start(stream);
                transform(gpu, stream, slot.buffer->data, lg, inverse, *shift);
                slot.kernel.end(stream);
                slot.job = &job;
            }
            for (size_t i = 0; i < slots; ++i) drain(i);
        }
        // Async frees precede the next memory admission on this device.
        gpu.sync();
        return 0;
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}

extern "C" int multi_stark_kzg_fft_from(int device, const fr_t* input,
                                         size_t input_count, fr_t* output,
                                         uint32_t lg, bool inverse, const fr_t* shift) {
    const FftJob job{input, nullptr, output, input_count, nullptr};
    return multi_stark_kzg_fft_batch(device, &job, 1, lg, inverse, shift, 1);
}

extern "C" int multi_stark_kzg_fft(int device, fr_t* values, uint32_t lg,
                                    bool inverse, const fr_t* shift) {
    if (lg > 31) return cudaErrorInvalidValue;
    return multi_stark_kzg_fft_from(device, values, size_t(1) << lg,
                                    values, lg, inverse, shift);
}

constexpr size_t DIVISION_TILE = 256;

__global__ void division_tiles(fr_t* data, size_t count, fr_t z, fr_t* carries,
                               bool apply_carry) {
    const size_t tile = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t begin = tile * DIVISION_TILE;
    if (begin >= count) return;
    const size_t end = min(begin + DIVISION_TILE, count);
    fr_t carry;
    carry.zero();
    if (apply_carry) carry = carries[tile];
    for (size_t i = end; i-- > begin;) {
        carry *= z;
        if (apply_carry) {
            data[i] += carry;
        } else {
            carry += data[i];
            data[i] = carry;
        }
    }
    if (!apply_carry) carries[tile] = carry;
}

extern "C" int multi_stark_kzg_evaluate_resident(int device, const fr_t* coefficients,
                                        const void* resident_pointer, size_t count,
                                        const fr_t* points, size_t point_count, fr_t* results) {
    try {
        if ((!coefficients && !resident_pointer) || !points || !results || count < 2 || point_count == 0)
            return cudaErrorInvalidValue;
        const auto& gpu = device_gpu(device);
        const auto* resident = static_cast<const ResidentPolynomial*>(resident_pointer);
        if (resident && (resident->device != device || resident->count != count))
            return cudaErrorInvalidValue;
        auto& transfer = transfer_lane(gpu);
        TransferStats stats;
        dev_ptr_t<fr_t> data(resident ? 0 : count, gpu);
        const fr_t* input = resident ? resident->data.data() : data.data();
        if (!resident)
            transfer.upload.upload(gpu, gpu, data, coefficients, count * sizeof(fr_t), stats);
        dev_ptr_t<fr_t> inputs(point_count, gpu), outputs(point_count, gpu);
        gpu.HtoD(&inputs[0], points, point_count);
        TimedKernel kernel;
        kernel.start(gpu);
        // The pinned multi-point kernel reuses shared power storage without a
        // block barrier. Single-point launches retain the coefficient upload
        // across points and avoid that shared-memory race.
        for (size_t i = 0; i < point_count; ++i)
            evaluate(&outputs[i], &inputs[i], 1, input, count, gpu);
        kernel.end(gpu);
        gpu.DtoH(results, &outputs[0], point_count);
        gpu.sync();
        transfer.upload.finish_upload(stats);
        kernel.collect(stats);
        stats.report(device, "evaluate", count);
        return 0;
    } catch (const cuda_error& e) { return failure(e.code(), e.what()); }
      catch (const std::exception& e) { return failure(cudaErrorUnknown, e.what()); }
}

extern "C" int multi_stark_kzg_evaluate(int device, const fr_t* coefficients,
                                        size_t count, const fr_t* points,
                                        size_t point_count, fr_t* results) {
    return multi_stark_kzg_evaluate_resident(device, coefficients, nullptr, count,
                                            points, point_count, results);
}

extern "C" int multi_stark_kzg_divide(int device, fr_t* values, size_t count,
                                      const fr_t* z) {
    // Keep transfer buffers alive through error-path synchronization.
    std::vector<fr_t> host;
    try {
        if (!values || !z || count < 2) return cudaErrorInvalidValue;
        const auto& gpu = device_gpu(device);
        dev_ptr_t<fr_t> data(count, gpu);
        auto& transfer = transfer_lane(gpu);
        TransferStats stats;
        transfer.upload.upload(gpu, gpu, data, values, count * sizeof(fr_t), stats);
        TimedKernel kernel;
        kernel.start(gpu);
        const size_t tiles = (count + DIVISION_TILE - 1) / DIVISION_TILE;
        dev_ptr_t<fr_t> carries(tiles, gpu);
        host.resize(tiles);
        const unsigned grid = static_cast<unsigned>((tiles + 127) / 128);
        division_tiles<<<grid, 128, 0, gpu>>>(&data[0], count, *z, &carries[0], false);
        CUDA_OK(cudaGetLastError());
        gpu.DtoH(host.data(), &carries[0], tiles);
        gpu.sync();
        // One carry per tile bounds host work and avoids any inter-block
        // synchronization or reads from partially overwritten coefficients.
        fr_t carry = 0;
        const fr_t step = *z ^ static_cast<unsigned>(DIVISION_TILE);
        for (size_t i = tiles; i-- > 0;) {
            const fr_t local = host[i];
            host[i] = carry;
            carry *= step;
            carry += local;
        }
        gpu.HtoD(&carries[0], host.data(), tiles);
        division_tiles<<<grid, 128, 0, gpu>>>(&data[0], count, *z, &carries[0], true);
        CUDA_OK(cudaGetLastError());
        kernel.end(gpu);
        transfer.download.download(gpu, gpu, values, &data[1], (count - 1) * sizeof(fr_t), stats);
        gpu.sync();
        transfer.upload.finish_upload(stats);
        kernel.collect(stats);
        stats.report(device, "divide", count);
        return 0;
    } catch (const cuda_error& e) {
        return failure(e.code(), e.what());
    } catch (const std::exception& e) {
        return failure(cudaErrorUnknown, e.what());
    }
}
