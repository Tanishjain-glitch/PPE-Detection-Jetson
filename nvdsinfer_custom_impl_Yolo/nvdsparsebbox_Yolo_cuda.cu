// nvdsparsebbox_Yolo_cuda.cu
// CUDA parser for YOLOv8, 6 classes ([cx,cy,w,h] + 6 scores per box)
// Class IDs: 0-'Gloves', 1-'Vest', 2-'goggles', 3-'helmet', 4-'mask', 5-'safety_shoe'

#include "nvdsinfer_custom_impl.h"
#include "nvtx3/nvToolsExt.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <iomanip>
#include <thrust/device_vector.h>

static const int NUM_CLASSES_YOLO = 6;
#define MAX_OUTPUT_BBOX_COUNT 8400
#define BLOCKSIZE 256

static thrust::device_vector<NvDsInferParseObjectInfo> d_objects(MAX_OUTPUT_BBOX_COUNT);

__device__ float clamp_cuda(float val, float minVal, float maxVal) {
    return fminf(maxVal, fmaxf(minVal, val));
}

__global__ void decodeYoloV8Kernel(
    const float* input,
    NvDsInferParseObjectInfo* output,
    int num_boxes,
    int num_classes,
    int netW,
    int netH,
    float threshold)
{
    int box_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (box_idx >= num_boxes) {
        return;
    }

    // Assuming data layout [cx0..cx8399, cy0.., w0.., h0.., 6 class scores x 8400]
    float cx = input[0 * num_boxes + box_idx];
    float cy = input[1 * num_boxes + box_idx];
    float w  = input[2 * num_boxes + box_idx];
    float h  = input[3 * num_boxes + box_idx];

    // Find the class with maximum score
    float maxScore = -1.0f;
    int maxClass = 0;
    for (int c = 0; c < num_classes; ++c) {
        float score = input[(4 + c) * num_boxes + box_idx];
        if (score > maxScore) {
            maxScore = score;
            maxClass = c;
        }
    }

    if (maxScore < threshold) {
        output[box_idx].detectionConfidence = 0.0f;
        return;
    }

    // Convert center format to corner format
    float x0 = cx - w / 2.0f;
    float y0 = cy - h / 2.0f;
    float x1 = cx + w / 2.0f;
    float y1 = cy + h / 2.0f;

    x0 = clamp_cuda(x0, 0.0f, (float)netW);
    y0 = clamp_cuda(y0, 0.0f, (float)netH);
    x1 = clamp_cuda(x1, 0.0f, (float)netW);
    y1 = clamp_cuda(y1, 0.0f, (float)netH);

    float box_w = x1 - x0;
    float box_h = y1 - y0;

    if (box_w < 1.0f || box_h < 1.0f) {
        output[box_idx].detectionConfidence = 0.0f;
        return;
    }

    output[box_idx].left = x0;
    output[box_idx].top = y0;
    output[box_idx].width = box_w;
    output[box_idx].height = box_h;
    output[box_idx].detectionConfidence = maxScore;
    output[box_idx].classId = maxClass;
}

extern "C" bool NvDsInferParseCustomYoloV8_cuda(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    nvtxRangePush("NvDsInferParseYoloV8_cuda");

    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: No output layers found\n";
        nvtxRangePop();
        return false;
    }

    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const float* data = (const float*)layer.buffer;
    
    int dimensions = 0;
    int num_boxes = 0;

    // Print info for debugging
    std::cerr << "[DEBUG] inferDims.numDims=" << layer.inferDims.numDims;
    for (int i = 0; i < layer.inferDims.numDims; ++i) {
        std::cerr << " d[" << i << "]=" << layer.inferDims.d[i];
    }
    std::cerr << std::endl;

    const int expected_dimensions = 4 + NUM_CLASSES_YOLO;

    if (layer.inferDims.numDims == 2) {
        int dim0 = layer.inferDims.d[0];
        int dim1 = layer.inferDims.d[1];

        if (dim0 == expected_dimensions) {
            dimensions = dim0;
            num_boxes = dim1;
        } else if (dim1 == expected_dimensions) {
            dimensions = dim1;
            num_boxes = dim0;
        } else {
            std::cerr << "ERROR: Unexpected 2D output shape for YOLOv8\n";
            nvtxRangePop();
            return false;
        }
    } else if (layer.inferDims.numDims == 3) {
        int dim1 = layer.inferDims.d[1];
        int dim2 = layer.inferDims.d[2];

        if (dim1 == expected_dimensions) {
            dimensions = dim1;
            num_boxes = dim2;
        } else if (dim2 == expected_dimensions) {
            dimensions = dim2;
            num_boxes = dim1;
        } else {
            std::cerr << "ERROR: Unexpected 3D output shape for YOLOv8\n";
            nvtxRangePop();
            return false;
        }
    } else {
        std::cerr << "ERROR: Unexpected output shape, numDims=" << layer.inferDims.numDims << std::endl;
        nvtxRangePop();
        return false;
    }

    std::cerr << "[CUDA YOLOv8-Parse] dimensions=" << dimensions 
              << " num_boxes=" << num_boxes
              << " netW=" << networkInfo.width 
              << " netH=" << networkInfo.height << std::endl;

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured) {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected: " << NUM_CLASSES_YOLO << std::endl;
    }

    if (num_boxes <= 0 || num_boxes > MAX_OUTPUT_BBOX_COUNT) {
        std::cerr << "ERROR: Invalid num_boxes=" << num_boxes << std::endl;
        nvtxRangePop();
        return false;
    }

    // Get minimum per-class threshold from config
    float min_threshold = 0.25f;
    if (!detectionParams.perClassPreclusterThreshold.empty()) {
        min_threshold = *std::min_element(
            detectionParams.perClassPreclusterThreshold.begin(),
            detectionParams.perClassPreclusterThreshold.end());
    }

    // Launch kernel
    const int blocksPerGrid = (num_boxes + BLOCKSIZE - 1) / BLOCKSIZE;
    std::cerr << "[CUDA] Launching kernel: blocks=" << blocksPerGrid 
              << " threads=" << BLOCKSIZE 
              << " total_threads=" << (blocksPerGrid * BLOCKSIZE) << std::endl;

    decodeYoloV8Kernel<<<blocksPerGrid, BLOCKSIZE>>>(
        data,
        thrust::raw_pointer_cast(d_objects.data()),
        num_boxes,
        NUM_CLASSES_YOLO,
        networkInfo.width,
        networkInfo.height,
        min_threshold
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        nvtxRangePop();
        return false;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        nvtxRangePop();
        return false;
    }

    // Copy results back to host
    std::vector<NvDsInferParseObjectInfo> h_objects(num_boxes);
    thrust::copy(d_objects.begin(), d_objects.begin() + num_boxes, h_objects.begin());

    objectList.clear();
    int valid_count = 0;
    for (int i = 0; i < num_boxes; ++i) {
        if (h_objects[i].detectionConfidence > 0.0f) {
            objectList.push_back(h_objects[i]);
            if (valid_count < 10) {
                std::cerr << std::fixed << std::setprecision(3)
                          << "[CUDA Detection " << valid_count << "] "
                          << "box_idx=" << i
                          << " class=" << h_objects[i].classId
                          << " score=" << h_objects[i].detectionConfidence
                          << " box(x,y,w,h)=("
                          << h_objects[i].left << "," << h_objects[i].top << ","
                          << h_objects[i].width << "," << h_objects[i].height << ")\n";
            }
            valid_count++;
        }
    }

    std::cerr << "[CUDA YOLOv8] Parsed " << objectList.size() 
              << " objects (threshold=" << min_threshold << ")" << std::endl;

    nvtxRangePop();
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV8_cuda);
