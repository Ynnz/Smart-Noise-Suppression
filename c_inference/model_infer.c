#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model_weights.h"

// ReLU activation
int16_t relu(int16_t x) {
    return x > 0 ? x : 0;
}

// 2D convolution with optional ReLU and padding=1
void conv2d(
    int16_t* input, int H, int W,
    int16_t* weights, int in_channels, int out_channels, int kernel_size,
    int16_t* bias,
    int16_t* output,
    int apply_relu
) {
    int pad = kernel_size / 2;
    int padded_H = H + 2 * pad;
    int padded_W = W + 2 * pad;

    // Allocate padded input
    int16_t* padded_input = (int16_t*)calloc(in_channels * padded_H * padded_W, sizeof(int16_t));
    if (!padded_input) {
        fprintf(stderr, "Error: malloc failed\n");
        return;
    }

    // Copy original input into padded buffer
    for (int c = 0; c < in_channels; ++c) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                int in_idx = (c * H + i) * W + j;
                int pad_idx = (c * padded_H + (i + pad)) * padded_W + (j + pad);
                padded_input[pad_idx] = input[in_idx];
            }
        }
    }

    int out_h = H;
    int out_w = W;

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int i = 0; i < out_h; ++i) {
            for (int j = 0; j < out_w; ++j) {
                int32_t acc = 0;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int ki = 0; ki < kernel_size; ++ki) {
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            int idx_in = ((ic * padded_H + (i + ki)) * padded_W + (j + kj));
                            int idx_wt = (((oc * in_channels + ic) * kernel_size + ki) * kernel_size + kj);
                            acc += (int32_t)padded_input[idx_in] * weights[idx_wt];
                        }
                    }
                }
                acc = acc >> 8;  // Scale back to Q7.8
                if (bias) acc += bias[oc];
                int16_t val = (int16_t)acc;
                if (apply_relu) val = relu(val);
                output[(oc * out_h + i) * out_w + j] = val;
            }
        }
    }

    free(padded_input);
}

// Inference pipeline
void model_infer(int16_t* input, int H, int W, int16_t* output) {
    int16_t* buf1 = (int16_t*)malloc(8 * H * W * sizeof(int16_t));
    int16_t* buf2 = (int16_t*)malloc(16 * H * W * sizeof(int16_t));
    int16_t* buf3 = (int16_t*)malloc(8 * H * W * sizeof(int16_t));

    conv2d(input, H, W, encoder_0_weight, 1, 8, 3, encoder_0_bias, buf1, 1);
    conv2d(buf1, H, W, encoder_2_weight, 8, 16, 3, encoder_2_bias, buf2, 1);
    conv2d(buf2, H, W, decoder_0_weight, 16, 8, 3, decoder_0_bias, buf3, 1);
    conv2d(buf3, H, W, decoder_2_weight, 8, 1, 3, decoder_2_bias, output, 0); // No ReLU on final layer

    free(buf1);
    free(buf2);
    free(buf3);
}