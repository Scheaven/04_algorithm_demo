#pragma once
void* trtCreate(const char* model_path, int batchSize, int INPUT_W, int INPUT_H, int INPUT_C, int OUTPUT_SIZE);
void trtRelease(void* handle);
void* trtDoInfer(void* handle, float* data, float* output);
void trtReleaseResult(void* result);

