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
/*
 * Minimal CPU vector-environment harness used by the standalone roboracer
 * shared library. The Python adapter owns scheduling; this layer only owns the
 * flat zero-copy buffers and dispatches reset/step across environments.
 */
#ifndef ROBORACER_VECENV_H
#define ROBORACER_VECENV_H

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* key;
    double value;
    void* ptr;
} DictItem;

typedef struct {
    DictItem* items;
    int size;
    int capacity;
} Dict;

typedef struct StaticVec {
    void* envs;
    int size;
    int total_agents;
    int buffers;
    int agents_per_buffer;
    int* buffer_env_starts;
    int* buffer_env_counts;
    void* observations;
    float* actions;
    float* rewards;
    float* terminals;
    void* gpu_observations;
    float* gpu_actions;
    float* gpu_rewards;
    float* gpu_terminals;
    void* streams;
    void* threading;
    int obs_size;
    int num_atns;
    int gpu;
} StaticVec;

/* Types used by the CPU-only CUDA stubs in shim.c. */
typedef struct CUstream_st* cudaStream_t;

static inline Dict* create_dict(int capacity) {
    Dict* dict = (Dict*)calloc(1, sizeof(Dict));
    assert(dict != NULL);
    dict->capacity = capacity;
    dict->items = (DictItem*)calloc((size_t)capacity, sizeof(DictItem));
    assert(dict->items != NULL);
    return dict;
}

static inline DictItem* dict_get_unsafe(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
}

static inline DictItem* dict_get(Dict* dict, const char* key) {
    DictItem* item = dict_get_unsafe(dict, key);
    if (item == NULL) {
        fprintf(stderr, "dict_get failed to find key: %s\n", key);
        abort();
    }
    return item;
}

static inline void dict_set(Dict* dict, const char* key, double value) {
    DictItem* item = dict_get_unsafe(dict, key);
    if (item != NULL) {
        item->value = value;
        return;
    }
    assert(dict->size < dict->capacity);
    item = &dict->items[dict->size++];
    item->key = key;
    item->value = value;
}

StaticVec* create_static_vec(
    int total_agents,
    int num_buffers,
    int gpu,
    Dict* vec_kwargs,
    Dict* env_kwargs);
void static_vec_reset(StaticVec* vec);
void static_vec_step(StaticVec* vec);
void static_vec_close(StaticVec* vec);
int get_obs_size(void);
int get_num_atns(void);

/*
 * binding.c defines the environment type and dimensions before including this
 * file. shim.c includes only the declarations above.
 */
#ifdef OBS_SIZE

void my_init(Env* env, Dict* kwargs);

StaticVec* create_static_vec(
    int total_agents,
    int num_buffers,
    int gpu,
    Dict* vec_kwargs,
    Dict* env_kwargs) {
    (void)vec_kwargs;
    assert(total_agents > 0);
    assert(num_buffers == 1);
    assert(gpu == 0);

    StaticVec* vec = (StaticVec*)calloc(1, sizeof(StaticVec));
    assert(vec != NULL);
    vec->size = total_agents;
    vec->total_agents = total_agents;
    vec->buffers = num_buffers;
    vec->agents_per_buffer = total_agents;
    vec->obs_size = OBS_SIZE;
    vec->num_atns = NUM_ATNS;

    vec->envs = calloc((size_t)total_agents, sizeof(Env));
    vec->observations =
        calloc((size_t)total_agents * OBS_SIZE, sizeof(float));
    vec->actions =
        (float*)calloc((size_t)total_agents * NUM_ATNS, sizeof(float));
    vec->rewards = (float*)calloc((size_t)total_agents, sizeof(float));
    vec->terminals = (float*)calloc((size_t)total_agents, sizeof(float));
    assert(vec->envs != NULL);
    assert(vec->observations != NULL);
    assert(vec->actions != NULL);
    assert(vec->rewards != NULL);
    assert(vec->terminals != NULL);

    vec->gpu_observations = vec->observations;
    vec->gpu_actions = vec->actions;
    vec->gpu_rewards = vec->rewards;
    vec->gpu_terminals = vec->terminals;

    Env* envs = (Env*)vec->envs;
    for (int i = 0; i < total_agents; i++) {
        envs[i].rng = (unsigned int)i;
        my_init(&envs[i], env_kwargs);
        assert(envs[i].num_agents == 1);
        envs[i].observations =
            (float*)vec->observations + (size_t)i * OBS_SIZE;
        envs[i].actions = vec->actions + (size_t)i * NUM_ATNS;
        envs[i].rewards = vec->rewards + i;
        envs[i].terminals = vec->terminals + i;
    }
    return vec;
}

void static_vec_reset(StaticVec* vec) {
    Env* envs = (Env*)vec->envs;
    memset(vec->rewards, 0, (size_t)vec->total_agents * sizeof(float));
    memset(vec->terminals, 0, (size_t)vec->total_agents * sizeof(float));
    for (int i = 0; i < vec->size; i++) {
        c_reset(&envs[i]);
    }
}

void static_vec_step(StaticVec* vec) {
    Env* envs = (Env*)vec->envs;
    memset(vec->rewards, 0, (size_t)vec->total_agents * sizeof(float));
    memset(vec->terminals, 0, (size_t)vec->total_agents * sizeof(float));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < vec->size; i++) {
        c_step(&envs[i]);
    }
}

void static_vec_close(StaticVec* vec) {
    if (vec == NULL) {
        return;
    }
    Env* envs = (Env*)vec->envs;
    for (int i = 0; i < vec->size; i++) {
        c_close(&envs[i]);
    }
    free(vec->envs);
    free(vec->observations);
    free(vec->actions);
    free(vec->rewards);
    free(vec->terminals);
    free(vec);
}

int get_obs_size(void) {
    return OBS_SIZE;
}

int get_num_atns(void) {
    return NUM_ATNS;
}

#endif
#endif
