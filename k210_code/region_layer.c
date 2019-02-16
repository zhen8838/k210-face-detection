/*
 * @Author: Zheng Qihang
 * @Date: 2019-02-15 21:13:36
 * @Last Modified by: Zheng Qihang
 * @Last Modified time: 2019-02-15 23:05:09
 */

#include "region_layer.h"
#include "act_ary.h"
#include "xy_offset.h"
#include <stdio.h>
#include <stdlib.h>
/**
 * @brief init the region layer
 *
 * @param rl    region_layer_t
 * @param task  kpu_task_t
 * @return int  0 success -1 error
 */
int region_layer_init(region_layer_t *rl, kpu_task_t *task) {
    kpu_layer_argument_t *last_layer= &task->layers[task->layers_length - 1];
    rl->input= (uint8_t *)task->dst;
    rl->threshold= .7;

    rl->layer_channel= last_layer->image_channel_num.data.o_ch_num + 1;
    rl->layer_height= last_layer->image_size.data.o_col_high + 1;
    rl->layer_width= last_layer->image_size.data.o_row_wid + 1;
    rl->grid_w= 1 / (float)rl->layer_width;
    rl->grid_h= 1 / (float)rl->layer_height;
    rl->offset= xy_offset;

    rl->input_len= rl->layer_channel * rl->layer_height * rl->layer_width;
    rl->output_len= rl->input_len;

    rl->scale= task->output_scale;
    rl->bias= task->output_bias;

    printf("region_layer_ch_num is %d\n", rl->layer_channel);
    printf("region_layer_l_w is %d\n", rl->layer_width);
    printf("region_layer_l_h is %d\n", rl->layer_height);
    printf("region_layer_out_len is %d\n", rl->output_len);

    rl->boxes_number= 0;
    rl->activate= sigmoid;

    rl->boxes= malloc(10 * sizeof(box_t)); // max 10 faces
    if (rl->boxes == NULL) { return -1; }
    rl->output= malloc(rl->output_len * sizeof(float));
    if (rl->output == NULL) { return -1; }

    return 0;
}

void region_layer_deinit(region_layer_t *rl) { free(rl->boxes); }

static void activate_array(region_layer_t *rl) {
    for (size_t i= 0; i < rl->input_len; i++) {
        rl->output[i]= rl->activate[rl->input[i]];
    }
}

static void xy_to_all(region_layer_t *rl) {
    for (size_t i= 0; i < rl->layer_height * rl->layer_width; i++) {
        rl->output[i]= rl->output[i] * rl->grid_w + rl->offset[i];
    }
    for (size_t i= rl->layer_height * rl->layer_width;
         i < 2 * rl->layer_height * rl->layer_width; i++) {
        rl->output[i]= rl->output[i] * rl->grid_h + rl->offset[i];
    }
}

static void label_to_box(region_layer_t *rl) {
    int box_cnt= 0;
    int layer_wh= rl->layer_height * rl->layer_width;
    uint16_t idx= 0;
    for (uint16_t i= (rl->layer_channel - 1) * layer_wh; i < rl->output_len; i++) {
        if (rl->output[i] >= rl->threshold) {
            if (box_cnt >= 10) {
                break;
            } else {
                idx= i;
                rl->boxes[box_cnt].prob= rl->output[idx];
                idx-= layer_wh;
                rl->boxes[box_cnt].h= rl->output[idx];
                idx-= layer_wh;
                rl->boxes[box_cnt].w= rl->output[idx];
                idx-= layer_wh;
                rl->boxes[box_cnt].y= rl->output[idx];
                idx-= layer_wh;
                rl->boxes[box_cnt].x= rl->output[idx];
                box_cnt++;
            }
        }
    }
    rl->boxes_number= box_cnt;
}

void run_region_layer(region_layer_t *rl) {
    activate_array(rl);
    xy_to_all(rl);
    label_to_box(rl);
}
