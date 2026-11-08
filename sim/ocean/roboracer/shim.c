/*
 * Structure adapted from PufferLib (https://github.com/PufferAI/PufferLib).
 * Copyright (c) 2022 PufferAI. Licensed under the MIT License:
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/* Flat C API over the static vecenv for loading via ctypes.
 *
 * Compiled WITHOUT OBS_SIZE: this translation unit sees vecenv.h types only;
 * the implementation lives in binding.c. Both files are linked into one .so.
 *
 * create_static_vec() allocates flat observations / actions / rewards /
 * terminals and points every env into per-agent slots.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vecenv.h"

/* ---- CUDA stubs -----------------------------------------------------------
 * vecenv.h's implementation references cudaMemcpy & friends. On the CPU path
 * (gpu=0, no create_static_threads) they are never CALLED, but ctypes dlopens
 * with RTLD_NOW, which fails on any undefined symbol — so define aborting
 * stubs. If one ever fires, someone requested the GPU path without CUDA
 * linked; die loudly instead of corrupting buffers. Signatures mirror the
 * extern declarations in vecenv.h (cudaError_t/cudaMemcpyKind are int there).
 */
#define CUDA_STUB(name) { fprintf(stderr, "roboracer shim: " #name \
    " called but this build has no CUDA (CPU-only shim)\n"); abort(); }
int cudaHostAlloc(void** p, size_t n, unsigned int f) CUDA_STUB(cudaHostAlloc)
int cudaMalloc(void** p, size_t n) CUDA_STUB(cudaMalloc)
int cudaMemcpy(void* d, const void* s, size_t n, int k) CUDA_STUB(cudaMemcpy)
int cudaMemcpyAsync(void* d, const void* s, size_t n, int k, cudaStream_t st) CUDA_STUB(cudaMemcpyAsync)
int cudaMemset(void* p, int v, size_t n) CUDA_STUB(cudaMemset)
int cudaFree(void* p) CUDA_STUB(cudaFree)
int cudaFreeHost(void* p) CUDA_STUB(cudaFreeHost)
int cudaSetDevice(int d) CUDA_STUB(cudaSetDevice)
int cudaDeviceSynchronize(void) CUDA_STUB(cudaDeviceSynchronize)
int cudaStreamSynchronize(cudaStream_t s) CUDA_STUB(cudaStreamSynchronize)
int cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int f) CUDA_STUB(cudaStreamCreateWithFlags)
int cudaStreamQuery(cudaStream_t s) CUDA_STUB(cudaStreamQuery)
const char* cudaGetErrorString(int e) CUDA_STUB(cudaGetErrorString)

/* num_envs is the number of single-agent environments. Keys are strdup'd
 * because Dict stores the pointer. Set ROBORACER_TRACK before create.
 */
void* rr_create(int num_envs, int n_kwargs, const char** keys, const double* values) {
    Dict* vec_kwargs = create_dict(4);
    dict_set(vec_kwargs, "total_agents", (double)num_envs);
    dict_set(vec_kwargs, "num_buffers", 1.0);

    Dict* env_kwargs = create_dict(n_kwargs + 1);
    for (int i = 0; i < n_kwargs; i++) {
        dict_set(env_kwargs, strdup(keys[i]), values[i]);
    }
    return create_static_vec(num_envs, 1, /*gpu=*/0, vec_kwargs, env_kwargs);
}

void rr_reset(void* vec) { static_vec_reset((StaticVec*)vec); }
void rr_step(void* vec)  { static_vec_step((StaticVec*)vec); }   /* OMP over envs */
void rr_close(void* vec) { static_vec_close((StaticVec*)vec); }

float* rr_observations(void* vec) { return (float*)((StaticVec*)vec)->observations; }
float* rr_actions(void* vec)      { return ((StaticVec*)vec)->actions; }
float* rr_rewards(void* vec)      { return ((StaticVec*)vec)->rewards; }
float* rr_terminals(void* vec)    { return ((StaticVec*)vec)->terminals; }

int rr_obs_size(void) { return get_obs_size(); }
int rr_num_atns(void) { return get_num_atns(); }
