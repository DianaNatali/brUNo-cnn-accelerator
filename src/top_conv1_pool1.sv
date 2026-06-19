`ifdef COCOTB_SIM
  `include "../src/parameters.svh"
`else
  `include "parameters.svh"
`endif

// -----------------------------------------------------------------------------
// top_conv1_pool1.sv
// Pipeline: Conv1 (24 kernels 3×3 + ReLU) → MaxPool1 (2×2) → LineBuffer
//
// Salidas planas para compatibilidad con Icarus 11:
//   pool1_out_o — KERNEL_NUM*BITS_Q4_6 bits, canal ch en [ch*11 +: 11]
//   lb_row0/1/2 — ídem
// -----------------------------------------------------------------------------

module top_conv1_pool1 #(
    parameter int IMG_WIDTH = 32
)(
    input  logic                       clk_i,
    input  logic                       nreset_i,

    input  logic                       start_cnn_i,
    input  logic                       px_rdy_i,
    input  logic [PIXEL_WIDTH_OUT-1:0] in_px_i,

    input  matrix_3x3_8bits            kernel_in,
    input  logic                       kernel_valid_i,
    output logic                       kernels_ready_o,

    output vector_8_Q4_6               conv1_out_o,
    output logic                       conv1_rdy_o,

    output logic [KERNEL_NUM*BITS_Q4_6-1:0] pool1_out_o,
    output logic                            pool1_rdy_o,

    output logic [KERNEL_NUM*BITS_Q4_6-1:0] lb_row0_o,
    output logic [KERNEL_NUM*BITS_Q4_6-1:0] lb_row1_o,
    output logic [KERNEL_NUM*BITS_Q4_6-1:0] lb_row2_o,
    output logic                             lb_px_rdy_o
);

    // -------------------------------------------------------------------------
    // Señales internas
    // -------------------------------------------------------------------------
    vector_8_Q4_6          conv1_out_w;
    logic                  conv1_rdy_w;

    logic [KERNEL_NUM-1:0] pool1_rdy_arr;
    logic                  pool1_rdy_w;
    logic [KERNEL_NUM-1:0] lb_rdy_arr;

    logic [BITS_Q4_6-1:0]  conv1_px [0:KERNEL_NUM-1];
    logic [BITS_Q4_6-1:0]  pool1_px [0:KERNEL_NUM-1];
    logic [BITS_Q4_6-1:0]  lb_r0    [0:KERNEL_NUM-1];
    logic [BITS_Q4_6-1:0]  lb_r1    [0:KERNEL_NUM-1];
    logic [BITS_Q4_6-1:0]  lb_r2    [0:KERNEL_NUM-1];

    assign pool1_rdy_w = &pool1_rdy_arr;
    assign pool1_rdy_o = pool1_rdy_w;
    assign lb_px_rdy_o = &lb_rdy_arr;

    // -------------------------------------------------------------------------
    // Conv Layer 1
    // -------------------------------------------------------------------------
    conv_layer conv1_inst (
        .clk_i          (clk_i),
        .nreset_i       (nreset_i),
        .start_cnn_i    (start_cnn_i),
        .px_rdy_i       (px_rdy_i),
        .in_value_i     (in_px_i),
        .kernel_in      (kernel_in),
        .kernel_valid_i (kernel_valid_i),
        .out_px_array   (conv1_out_w),
        .px_rdy_o       (conv1_rdy_w),
        .kernels_ready_o(kernels_ready_o)
    );

    assign conv1_out_o = conv1_out_w;
    assign conv1_rdy_o = conv1_rdy_w;

    assign conv1_px[0]  = conv1_out_w.p0;
    assign conv1_px[1]  = conv1_out_w.p1;
    assign conv1_px[2]  = conv1_out_w.p2;
    assign conv1_px[3]  = conv1_out_w.p3;
    assign conv1_px[4]  = conv1_out_w.p4;
    assign conv1_px[5]  = conv1_out_w.p5;
    assign conv1_px[6]  = conv1_out_w.p6;
    assign conv1_px[7]  = conv1_out_w.p7;
    assign conv1_px[8]  = conv1_out_w.p8;
    assign conv1_px[9]  = conv1_out_w.p9;
    assign conv1_px[10] = conv1_out_w.p10;
    assign conv1_px[11] = conv1_out_w.p11;
    assign conv1_px[12] = conv1_out_w.p12;
    assign conv1_px[13] = conv1_out_w.p13;
    assign conv1_px[14] = conv1_out_w.p14;
    assign conv1_px[15] = conv1_out_w.p15;
    assign conv1_px[16] = conv1_out_w.p16;
    assign conv1_px[17] = conv1_out_w.p17;
    assign conv1_px[18] = conv1_out_w.p18;
    assign conv1_px[19] = conv1_out_w.p19;
    assign conv1_px[20] = conv1_out_w.p20;
    assign conv1_px[21] = conv1_out_w.p21;
    assign conv1_px[22] = conv1_out_w.p22;
    assign conv1_px[23] = conv1_out_w.p23;

    // -------------------------------------------------------------------------
    // MaxPool1: un max_pooling_ctr por canal
    // out_px_o es matrix_2x2_Q4_6 (4*BITS_Q4_6 = 44 bits packed).
    // Se conecta a wire flat y se extrae vector0.p0 = bits [BITS_Q4_6-1:0].
    // Esto evita el crash de Icarus al indexar array de structs en generate.
    // -------------------------------------------------------------------------
    genvar ch;
    generate
        for (ch = 0; ch < KERNEL_NUM; ch = ch + 1) begin : gen_pool
            wire [4*BITS_Q4_6-1:0] pool_flat;

            max_pooling_ctr pool_inst (
                .clk_i      (clk_i),
                .nreset_i   (nreset_i),
                .start_mp_i (conv1_rdy_w),
                .px_rdy_i   (conv1_rdy_w),
                .in_value_i (conv1_px[ch]),
                .out_px_o   (pool_flat),
                .px_rdy_o   (pool1_rdy_arr[ch])
            );

            // matrix_2x2_Q4_6 packed: {vector1.p1, vector1.p0, vector0.p1, vector0.p0}
            // vector0.p0 está en bits [BITS_Q4_6-1:0]
            assign pool1_px[ch] = pool_flat[BITS_Q4_6-1:0];
            assign pool1_out_o[ch*BITS_Q4_6 +: BITS_Q4_6] = pool1_px[ch];
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Line Buffer: uno por canal, ancho IMG_WIDTH/2
    // -------------------------------------------------------------------------
    generate
        for (ch = 0; ch < KERNEL_NUM; ch = ch + 1) begin : gen_lb
            line_buffer #(
                .IMG_WIDTH  (IMG_WIDTH / 2),
                .DATA_WIDTH (BITS_Q4_6)
            ) lb_inst (
                .clk_i     (clk_i),
                .nreset_i  (nreset_i),
                .px_rdy_i  (pool1_rdy_w),
                .in_px_i   (pool1_px[ch]),
                .out_row0_o(lb_r0[ch]),
                .out_row1_o(lb_r1[ch]),
                .out_row2_o(lb_r2[ch]),
                .px_rdy_o  (lb_rdy_arr[ch])
            );

            assign lb_row0_o[ch*BITS_Q4_6 +: BITS_Q4_6] = lb_r0[ch];
            assign lb_row1_o[ch*BITS_Q4_6 +: BITS_Q4_6] = lb_r1[ch];
            assign lb_row2_o[ch*BITS_Q4_6 +: BITS_Q4_6] = lb_r2[ch];
        end
    endgenerate

endmodule