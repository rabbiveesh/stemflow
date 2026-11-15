// Minimal LiteRT 2.x CompiledModel runner: times a .tflite on the GPU accelerator (ML Drift).
// usage: run MODEL PRECISION(0 default,1 fp16,2 fp32) RUNS [IN.f32 OUT.f32] [extra toml]
// env ACCEL: gpu (default), npu, npu+gpu, npu+cpu, cpu. The NPU needs the two .so files of
// litert-npu-runtime-google-tensor beside the binary and /vendor/lib64 in LD_LIBRARY_PATH.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#define OK(x) do { LiteRtStatus s_ = (x); if (s_ != kLiteRtStatusOk) { fprintf(stderr, "FAILED %s -> %d\n", #x, (int)s_); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char** argv) {
  const char* path = argv[1]; int precision = atoi(argv[2]); int runs = atoi(argv[3]);
  const char* accel = getenv("ACCEL") ? getenv("ACCEL") : "gpu";
  int hw = 0;
  if (strstr(accel, "gpu")) hw |= kLiteRtHwAcceleratorGpu;
  if (strstr(accel, "npu")) hw |= kLiteRtHwAcceleratorNpu;
  if (strstr(accel, "cpu")) hw |= kLiteRtHwAcceleratorCpu;
  LiteRtEnvOptionTag tags[3] = {kLiteRtEnvOptionTagRuntimeLibraryDir, kLiteRtEnvOptionTagCompilerPluginLibraryDir, kLiteRtEnvOptionTagDispatchLibraryDir};
  LiteRtEnvOption eo[3];
  for (int i = 0; i < 3; i++) { eo[i].tag = tags[i]; eo[i].value.type = kLiteRtAnyTypeString; eo[i].value.str_value = "."; }
  LiteRtEnvironment env; OK(LiteRtCreateEnvironment(hw & kLiteRtHwAcceleratorNpu ? 3 : 1, eo, &env));
  LiteRtModel model; OK(LiteRtCreateModelFromFile(env, path, &model));
  LiteRtOptions opts; OK(LiteRtCreateOptions(&opts));
  OK(LiteRtSetOptionsHardwareAccelerators(opts, hw));
  char toml[512]; snprintf(toml, sizeof toml, "precision = %d\n%s\n", precision, argc > 6 ? argv[6] : "");
  if ((hw & kLiteRtHwAcceleratorGpu) && (precision || argc > 6)) { LiteRtOpaqueOptions oo; OK(LiteRtCreateOpaqueOptions("gpu_options", strdup(toml), free, &oo)); OK(LiteRtAddOpaqueOptions(opts, oo)); }
  double t0 = now();
  LiteRtCompiledModel cm; OK(LiteRtCreateCompiledModel(env, model, opts, &cm));
  bool full = false; LiteRtCompiledModelIsFullyAccelerated(cm, &full);
  bool noncpu = false; LiteRtCompiledModelIsNonCpuFullyAccelerated(cm, &noncpu);
  fprintf(stderr, "compiled in %.2f s (%s), fully accelerated: %d, without CPU: %d\n", now() - t0, accel, (int)full, (int)noncpu);
  LiteRtSignature sig; OK(LiteRtGetModelSignature(model, 0, &sig));
  LiteRtParamIndex ni, no; OK(LiteRtGetNumSignatureInputs(sig, &ni)); OK(LiteRtGetNumSignatureOutputs(sig, &no));
  LiteRtTensorBuffer in[8], out[8]; size_t in_bytes[8], out_bytes[8];
  for (int k = 0; k < 2; k++) for (LiteRtParamIndex i = 0; i < (k ? no : ni); i++) {
    LiteRtTensor t; LiteRtRankedTensorType tt; LiteRtTensorBufferRequirements req; size_t sz;
    if (k) { OK(LiteRtGetSignatureOutputTensorByIndex(sig, i, &t)); OK(LiteRtGetCompiledModelOutputBufferRequirements(cm, 0, i, &req)); }
    else { OK(LiteRtGetSignatureInputTensorByIndex(sig, i, &t)); OK(LiteRtGetCompiledModelInputBufferRequirements(cm, 0, i, &req)); }
    OK(LiteRtGetRankedTensorType(t, &tt)); OK(LiteRtGetTensorBufferRequirementsBufferSize(req, &sz));
    size_t n = 4; fprintf(stderr, "%s %d: [", k ? "out" : "in", (int)i);
    for (unsigned d = 0; d < tt.layout.rank; d++) { n *= tt.layout.dimensions[d]; fprintf(stderr, "%d ", tt.layout.dimensions[d]); }
    fprintf(stderr, "] buffer %zu bytes\n", sz);
    OK(LiteRtCreateManagedTensorBufferFromRequirements(env, &tt, req, k ? &out[i] : &in[i]));
    if (k) out_bytes[i] = n; else in_bytes[i] = n;
  }
  FILE* fin = argc > 4 ? fopen(argv[4], "rb") : NULL; unsigned seed = 12345;
  for (LiteRtParamIndex i = 0; i < ni; i++) {
    void* p; OK(LiteRtLockTensorBuffer(in[i], &p, kLiteRtTensorBufferLockModeWrite));
    if (fin) { if (fread(p, 1, in_bytes[i], fin) != in_bytes[i]) { fprintf(stderr, "short input\n"); return 1; } }
    else { float* f = p; for (size_t j = 0; j < in_bytes[i] / 4; j++) { seed = seed * 1664525u + 1013904223u; f[j] = ((seed >> 8) / 16777216.0f - 0.5f) * 0.2f; } }
    OK(LiteRtUnlockTensorBuffer(in[i]));
  }
  for (int r = 0; r < runs; r++) {
    double t = now();
    OK(LiteRtRunCompiledModel(cm, 0, ni, in, no, out));
    for (LiteRtParamIndex i = 0; i < no; i++) { void* p; OK(LiteRtLockTensorBuffer(out[i], &p, kLiteRtTensorBufferLockModeRead)); OK(LiteRtUnlockTensorBuffer(out[i])); }
    fprintf(stderr, "run %d: %.3f s\n", r, now() - t);
  }
  if (argc > 5) { FILE* fo = fopen(argv[5], "wb"); for (LiteRtParamIndex i = 0; i < no; i++) { void* p; OK(LiteRtLockTensorBuffer(out[i], &p, kLiteRtTensorBufferLockModeRead)); fwrite(p, 1, out_bytes[i], fo); OK(LiteRtUnlockTensorBuffer(out[i])); } fclose(fo); }
  return 0;
}
