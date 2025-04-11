#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model_weights.h"

// Fixed-point multiply (Q7.8)
int16_t qmul(int16_t a, int16_t b) {
    return (int16_t)(((int32_t)a * b) >> 8);
}

// ReLU activation
int16_t relu(int16_t x) {
    return x > 0 ? x : 0;
}

// Declare conv2d before main
void conv2d(
    int16_t* input, int H, int W,
    int16_t* weights, int in_channels, int out_channels, int kernel_size,
    int16_t* bias,
    int16_t* output
);

// 2D convolution implementation
void conv2d(
    int16_t* input, int H, int W,
    int16_t* weights, int in_channels, int out_channels, int kernel_size,
    int16_t* bias,
    int16_t* output
) {
    int out_h = H - kernel_size + 1;
    int out_w = W - kernel_size + 1;
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int i = 0; i < out_h; ++i) {
            for (int j = 0; j < out_w; ++j) {
                int32_t acc = 0;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int ki = 0; ki < kernel_size; ++ki) {
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            int idx_in = ((ic * H + (i+ki)) * W + (j+kj));
                            int idx_wt = (((oc * in_channels + ic) * kernel_size + ki) * kernel_size + kj);
                            acc += (int32_t)input[idx_in] * weights[idx_wt];
                        }
                    }
                }
                acc = acc >> 8; // Scale back down
                output[(oc * out_h + i) * out_w + j] = relu((int16_t)acc);
            }
        }
    }
}

int main() {
    // Dummy 4x4 input (1 channel)
    int16_t input[4 * 4] = {
        256, 512, 768, 1024,
        256, 512, 768, 1024,
        256, 512, 768, 1024,
        256, 512, 768, 1024
    };

    // 1 filter, 1 input channel, 3x3 kernel → total 9 weights
    int16_t weights[9] = {
        256, 0, -256,
        0, 256, 0,
        -256, 0, 256
    };

    // No bias for simplicity
    int16_t bias[1] = {0};

    // Output will be 2x2 (since 4 - 3 + 1 = 2)
    int16_t output[2 * 2] = {0};

    conv2d(input, 4, 4, weights, 1, 1, 3, bias, output);

    printf("Output:\n");
    for (int i = 0; i < 2 * 2; ++i) {
        printf("%d ", output[i]);
        if ((i + 1) % 2 == 0) printf("\n");
    }

    return 0;
}