`ifdef COCOTB_SIM
  `include "../src/parameters.svh"
`else
  `include "parameters.svh"
  `include "conv_control.sv"
  `include "conv_core.sv"
`endif

module conv_layer (
    input clk_i,
    input nreset_i,
    input start_cnn_i,  
    input px_rdy_i,
    input [PIXEL_WIDTH_OUT-1:0] in_value_i,
    
    input matrix_3x3_8bits kernel_in,
      
    input kernel_valid_i,  
    output vector_8_Q4_6 out_px_array,
    output logic px_rdy_o,
    output logic kernels_ready_o 
);

    matrix_3x3_8bits kernel_values [KERNEL_NUM-1:0];  
    logic [KERNEL_COUNTER_BITS:0] kernel_counter;

    logic [KERNEL_NUM-1:0] px_rdy_out;

    logic [BITS_Q4_6-1:0] out_array[KERNEL_NUM-1:0];

    always_ff @(posedge clk_i) begin
        if (!nreset_i) begin
            integer i; //To -do
            for (i = 0; i < KERNEL_NUM; i = i + 1) begin
                kernel_values[i] <= 'b0;
            end
            kernel_counter <= 'b0;  
        end else if (kernel_valid_i && kernel_counter < KERNEL_NUM) begin
            kernel_values[kernel_counter] <= kernel_in; 
            kernel_counter <= kernel_counter + 'b1;
        end else begin
            kernel_counter <= 'b0;  
        end
    end

    assign kernels_ready_o = (kernel_counter == KERNEL_NUM);

    genvar i;
    generate
        for (i = 0; i < KERNEL_NUM; i = i + 1) begin : gen_conv
            conv_control conv_inst (
                .clk_i(clk_i),
                .nreset_i(nreset_i),
                .kernel_i(kernel_values[i]),  
                .start_cnn_i(start_cnn_i),  
                .px_rdy_i(px_rdy_i),
                .in_value_i(in_value_i),
                .out_px_o(out_array[i]), 
                .px_rdy_o(px_rdy_out[i])  
            );
        end
    endgenerate

    assign px_rdy_o = &px_rdy_out;
    
    assign out_px_array.p0 = out_array[0];
    assign out_px_array.p1 = out_array[1];
    assign out_px_array.p2 = out_array[2];
    assign out_px_array.p3 = out_array[3];
    assign out_px_array.p4 = out_array[4];
    assign out_px_array.p5 = out_array[5];
    assign out_px_array.p6 = out_array[6];
    assign out_px_array.p7 = out_array[7];
    assign out_px_array.p8 = out_array[8];
    assign out_px_array.p9 = out_array[9];
    assign out_px_array.p10 = out_array[10];
    assign out_px_array.p11 = out_array[11];
    assign out_px_array.p12 = out_array[12];
    assign out_px_array.p13 = out_array[13];
    assign out_px_array.p14 = out_array[14];
    assign out_px_array.p15 = out_array[15];
    assign out_px_array.p16 = out_array[16];
    assign out_px_array.p17 = out_array[17];
    assign out_px_array.p18 = out_array[18];
    assign out_px_array.p19 = out_array[19];
    assign out_px_array.p20 = out_array[20];
    assign out_px_array.p21 = out_array[21];
    assign out_px_array.p22 = out_array[22];
    assign out_px_array.p23 = out_array[23];
endmodule
