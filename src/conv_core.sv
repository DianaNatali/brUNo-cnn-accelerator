`ifdef COCOTB_SIM
  `include "../src/parameters.svh"
`else
  `include "parameters.svh"
`endif

`ifndef CONV_CORE_INCLUDED
`define CONV_CORE_INCLUDED

module conv_core(
  input matrix_3x3_8bits input_data,                    // Input 3x3 pixel matrix in Q1.6
  input matrix_3x3_8bits kernel,                        // Input Kernel 3x3 in Q1.6
  output [BITS_Q4_6-1:0] out_conv_o                     // Output 11 bits in Q4.6 + sign
);

  // Each 8bit x 8bit multiplication = 16 bits
  // 9 accumulations of 16 bits -> requires 16 + ceil(log2(9)) = 16 + 4 = 20 bits
  localparam MULT_BITS  = 2 * PIXEL_WIDTH_OUT;           // 16 bits
  localparam ACCUM_BITS = MULT_BITS + 4;                 // 20 bits (headroom for 9 additions)
  localparam ROUND_BIT  = 1 << (FRAC_BITS - 1);         // 0.5 LSB for rounding

  // Intermediate wires — Icarus does not support $signed() directly on struct members
  wire signed [PIXEL_WIDTH_OUT-1:0] px0, px1, px2, px3, px4, px5, px6, px7, px8;
  wire signed [PIXEL_WIDTH_OUT-1:0] k0,  k1,  k2,  k3,  k4,  k5,  k6,  k7,  k8;

  assign px0 = input_data.vector0.p0;  assign k0 = kernel.vector0.p0;
  assign px1 = input_data.vector0.p1;  assign k1 = kernel.vector0.p1;
  assign px2 = input_data.vector0.p2;  assign k2 = kernel.vector0.p2;
  assign px3 = input_data.vector1.p0;  assign k3 = kernel.vector1.p0;
  assign px4 = input_data.vector1.p1;  assign k4 = kernel.vector1.p1;
  assign px5 = input_data.vector1.p2;  assign k5 = kernel.vector1.p2;
  assign px6 = input_data.vector2.p0;  assign k6 = kernel.vector2.p0;
  assign px7 = input_data.vector2.p1;  assign k7 = kernel.vector2.p1;
  assign px8 = input_data.vector2.p2;  assign k8 = kernel.vector2.p2;

  wire signed [ACCUM_BITS-1:0] conv_full;
  wire signed [BITS_Q4_6-1:0]  conv_result;

  // Accumulate at full precision — single shift at the end
  assign conv_full = (px0 * k0) + (px1 * k1) + (px2 * k2) +
                     (px3 * k3) + (px4 * k4) + (px5 * k5) +
                     (px6 * k6) + (px7 * k7) + (px8 * k8);

  // Round-to-nearest: add 0.5 LSB before truncation to reduce quantization error
  assign conv_result = (conv_full + ROUND_BIT) >> FRAC_BITS;

  // ReLU
  assign out_conv_o = (conv_result[BITS_Q4_6-1] == 1'b1) ? {BITS_Q4_6{1'b0}} : conv_result;

endmodule
`endif