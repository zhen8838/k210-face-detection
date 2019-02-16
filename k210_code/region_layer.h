/*
 * @Author: Zheng Qihang
 * @Date: 2019-02-15 21:13:26
 * @Last Modified by: Zheng Qihang
 * @Last Modified time: 2019-02-15 22:12:22
 */
#ifndef _REGION_LAYER
#define _REGION_LAYER

#include "kpu.h"
#include <stdint.h>

typedef struct {
    float x;
    float y;
    float w;
    float h;
    float prob;
} box_t;

typedef struct {
    float threshold;
    uint32_t layer_channel;
    uint32_t layer_width;
    uint32_t layer_height;
    uint32_t boxes_number;
    float grid_w;
    float grid_h;
    float *offset;
    float *output;
    uint32_t output_len;

    float scale;
    float bias;
    box_t *boxes;
    uint8_t *input;
    uint32_t input_len;
    float *activate;
} region_layer_t;

extern int region_layer_init(region_layer_t *rl, kpu_task_t *task);
extern void region_layer_deinit(region_layer_t *rl);
extern void run_region_layer(region_layer_t *rl);

#endif // _REGION_LAYER
