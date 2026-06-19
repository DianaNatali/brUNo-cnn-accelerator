`ifdef COCOTB_SIM
  `include "../src/parameters.svh"
`else
  `include "parameters.svh"
`endif

module maxpooling_2x2 (
    input  matrix_4x4_Q4_6 input_data,
    output matrix_2x2_Q4_6 output_data
);

    // Icarus 11 no acepta $signed() sobre miembros de struct directamente.
    // Mismo patrón que conv_core.sv: extraer a wire signed primero.
    wire signed [BITS_Q4_6-1:0] v0p0, v0p1, v0p2, v0p3;
    wire signed [BITS_Q4_6-1:0] v1p0, v1p1, v1p2, v1p3;
    wire signed [BITS_Q4_6-1:0] v2p0, v2p1, v2p2, v2p3;
    wire signed [BITS_Q4_6-1:0] v3p0, v3p1, v3p2, v3p3;

    assign v0p0 = input_data.vector0.p0; assign v0p1 = input_data.vector0.p1;
    assign v0p2 = input_data.vector0.p2; assign v0p3 = input_data.vector0.p3;
    assign v1p0 = input_data.vector1.p0; assign v1p1 = input_data.vector1.p1;
    assign v1p2 = input_data.vector1.p2; assign v1p3 = input_data.vector1.p3;
    assign v2p0 = input_data.vector2.p0; assign v2p1 = input_data.vector2.p1;
    assign v2p2 = input_data.vector2.p2; assign v2p3 = input_data.vector2.p3;
    assign v3p0 = input_data.vector3.p0; assign v3p1 = input_data.vector3.p1;
    assign v3p2 = input_data.vector3.p2; assign v3p3 = input_data.vector3.p3;

    logic signed [BITS_Q4_6-1:0] tmp00, tmp01, tmp10, tmp11;

    always_comb begin
        // Cuadrante superior-izquierdo → output_data.vector0.p0
        tmp00 = (v0p0 > v0p1) ? v0p0 : v0p1;
        tmp00 = (tmp00 > v1p0) ? tmp00 : v1p0;
        tmp00 = (tmp00 > v1p1) ? tmp00 : v1p1;
        output_data.vector0.p0 = tmp00;

        // Cuadrante superior-derecho → output_data.vector0.p1
        tmp01 = (v0p2 > v0p3) ? v0p2 : v0p3;
        tmp01 = (tmp01 > v1p2) ? tmp01 : v1p2;
        tmp01 = (tmp01 > v1p3) ? tmp01 : v1p3;
        output_data.vector0.p1 = tmp01;

        // Cuadrante inferior-izquierdo → output_data.vector1.p0
        tmp10 = (v2p0 > v2p1) ? v2p0 : v2p1;
        tmp10 = (tmp10 > v3p0) ? tmp10 : v3p0;
        tmp10 = (tmp10 > v3p1) ? tmp10 : v3p1;
        output_data.vector1.p0 = tmp10;

        // Cuadrante inferior-derecho → output_data.vector1.p1
        tmp11 = (v2p2 > v2p3) ? v2p2 : v2p3;
        tmp11 = (tmp11 > v3p2) ? tmp11 : v3p2;
        tmp11 = (tmp11 > v3p3) ? tmp11 : v3p3;
        output_data.vector1.p1 = tmp11;
    end

endmodule