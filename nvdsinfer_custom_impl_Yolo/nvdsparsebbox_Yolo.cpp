#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <iomanip>
#include "nvdsinfer_custom_impl.h"

static const int NUM_CLASSES_YOLO = 6;  // <-- Update here

float clamp(const float val, const float minVal, const float maxVal) {
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

static bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: No output layer found\n";
        return false;
    }

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    float* data = (float*)layer.buffer;
    int dimensions = 0;
    int num_boxes = 0;

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
            return false;
        }
    } else {
        std::cerr << "ERROR: Unexpected output shape\n";
        return false;
    }

    std::cerr << "[YOLOv8-Parse] dimensions=" << dimensions 
              << " num_boxes=" << num_boxes
              << " netW=" << networkInfo.width 
              << " netH=" << networkInfo.height << std::endl;

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured) {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected: " << NUM_CLASSES_YOLO << std::endl;
    }

    float min_threshold = 0.25f;
    if (!detectionParams.perClassPreclusterThreshold.empty()) {
        min_threshold = *std::min_element(
            detectionParams.perClassPreclusterThreshold.begin(),
            detectionParams.perClassPreclusterThreshold.end());
    }

    objectList.clear();
    int valid_count = 0;

    for (int box_idx = 0; box_idx < num_boxes; ++box_idx) {
        float cx = data[0 * num_boxes + box_idx];
        float cy = data[1 * num_boxes + box_idx];
        float w  = data[2 * num_boxes + box_idx];
        float h  = data[3 * num_boxes + box_idx];

        float maxScore = -1.0f;
        int maxIndex = 0;
        for (int c = 0; c < NUM_CLASSES_YOLO; ++c) {
            float score = data[(4 + c) * num_boxes + box_idx];
            if (score > maxScore) {
                maxScore = score;
                maxIndex = c;
            }
        }

        float class_threshold = min_threshold;
        if (maxIndex < (int)detectionParams.perClassPreclusterThreshold.size()) {
            class_threshold = detectionParams.perClassPreclusterThreshold[maxIndex];
        }

        if (maxScore < class_threshold)
            continue;

        float x0 = cx - w / 2.0f;
        float y0 = cy - h / 2.0f;
        float x1 = cx + w / 2.0f;
        float y1 = cy + h / 2.0f;
        x0 = clamp(x0, 0.0f, static_cast<float>(networkInfo.width));
        y0 = clamp(y0, 0.0f, static_cast<float>(networkInfo.height));
        x1 = clamp(x1, 0.0f, static_cast<float>(networkInfo.width));
        y1 = clamp(y1, 0.0f, static_cast<float>(networkInfo.height));

        float box_w = x1 - x0;
        float box_h = y1 - y0;
        if (box_w < 1.0f || box_h < 1.0f)
            continue;

        NvDsInferParseObjectInfo obj;
        obj.left = x0;
        obj.top = y0;
        obj.width = box_w;
        obj.height = box_h;
        obj.detectionConfidence = maxScore;
        obj.classId = maxIndex;
        objectList.push_back(obj);

        if (valid_count < 10) {
            std::cerr << "[Detection " << valid_count << "] "
                      << "box_idx=" << box_idx
                      << " class=" << maxIndex 
                      << " score=" << std::fixed << std::setprecision(3) << maxScore
                      << " box(x,y,w,h)=(" << x0 << "," << y0 << "," << box_w << "," << box_h << ")"
                      << std::endl;
        }
        valid_count++;
    }

    std::cerr << "[YOLOv8-Parse] Total detections: " << objectList.size() 
              << " (threshold=" << min_threshold << ")" << std::endl;

    return true;
}

extern "C" bool NvDsInferParseCustomYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV8(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV8);
