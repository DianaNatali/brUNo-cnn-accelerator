`ifdef COCOTB_SIM
  `include "../src/parameters.svh"
`else
  `include "parameters.svh"
`endif

module max_pooling_ctr (
    input  logic clk_i,
    input  logic nreset_i,

    input  logic start_mp_i,
    input  logic px_rdy_i,
    input  logic [BITS_Q4_6-1:0] in_value_i,   // Q4.6 11 bits

    output matrix_2x2_Q4_6 out_px_o,
    output logic            px_rdy_o
);

    localparam IDLE         = 2'd0;
    localparam FIRST_MATRIX = 2'd1;
    localparam NEXT_MATRIX  = 2'd2;

    reg [1:0] fsm_state, next;

    logic [3:0]                     counter_pool;
    logic [MAX_RESOLUTION_BITS-1:0] counter_pixels;
    logic                           px_ready;

    matrix_4x4_Q4_6 matrix_in_values;
    matrix_2x2_Q4_6 pool_result;
    matrix_2x2_Q4_6 out_reg;

    maxpooling_2x2 pool_core (
        .input_data (matrix_in_values),
        .output_data(pool_result)
    );

    always_ff @(posedge clk_i) begin
        if (!nreset_i) fsm_state <= IDLE;
        else           fsm_state <= next;
    end

    always_comb begin
        case (fsm_state)
            IDLE:         next = start_mp_i           ? FIRST_MATRIX : IDLE;
            FIRST_MATRIX: next = (counter_pixels == 1) ? NEXT_MATRIX  : FIRST_MATRIX;
            NEXT_MATRIX:  next = start_mp_i           ? NEXT_MATRIX  : IDLE;
            default:      next = IDLE;
        endcase
    end

    always_ff @(posedge clk_i) begin
        if (!nreset_i) begin
            counter_pool     <= '0;
            counter_pixels   <= '0;
            px_ready         <= '0;
            matrix_in_values <= '0;
        end else begin
            px_ready <= '0;

            case (next)
                IDLE: begin
                    counter_pool     <= '0;
                    counter_pixels   <= '0;
                    matrix_in_values <= '0;
                end

                FIRST_MATRIX: begin
                    if (px_rdy_i) begin
                        case (counter_pool)
                            4'd0:  matrix_in_values.vector0.p0 <= in_value_i;
                            4'd1:  matrix_in_values.vector0.p1 <= in_value_i;
                            4'd2:  matrix_in_values.vector0.p2 <= in_value_i;
                            4'd3:  matrix_in_values.vector0.p3 <= in_value_i;
                            4'd4:  matrix_in_values.vector1.p0 <= in_value_i;
                            4'd5:  matrix_in_values.vector1.p1 <= in_value_i;
                            4'd6:  matrix_in_values.vector1.p2 <= in_value_i;
                            4'd7:  matrix_in_values.vector1.p3 <= in_value_i;
                            4'd8:  matrix_in_values.vector2.p0 <= in_value_i;
                            4'd9:  matrix_in_values.vector2.p1 <= in_value_i;
                            4'd10: matrix_in_values.vector2.p2 <= in_value_i;
                            4'd11: matrix_in_values.vector2.p3 <= in_value_i;
                            4'd12: matrix_in_values.vector3.p0 <= in_value_i;
                            4'd13: matrix_in_values.vector3.p1 <= in_value_i;
                            4'd14: matrix_in_values.vector3.p2 <= in_value_i;
                            4'd15: matrix_in_values.vector3.p3 <= in_value_i;
                            default: ;
                        endcase
                        if (counter_pool == 4'd15) begin
                            counter_pool   <= '0;
                            counter_pixels <= counter_pixels + 1'b1;
                            px_ready       <= '1;
                        end else begin
                            counter_pool <= counter_pool + 1'b1;
                        end
                    end
                end

                NEXT_MATRIX: begin
                    if (px_rdy_i) begin
                        case (counter_pool)
                            3'd0: begin
                                matrix_in_values.vector0    <= matrix_in_values.vector2;
                                matrix_in_values.vector1    <= matrix_in_values.vector3;
                                matrix_in_values.vector2.p0 <= in_value_i;
                            end
                            3'd1: matrix_in_values.vector2.p1 <= in_value_i;
                            3'd2: matrix_in_values.vector2.p2 <= in_value_i;
                            3'd3: matrix_in_values.vector2.p3 <= in_value_i;
                            3'd4: matrix_in_values.vector3.p0 <= in_value_i;
                            3'd5: matrix_in_values.vector3.p1 <= in_value_i;
                            3'd6: matrix_in_values.vector3.p2 <= in_value_i;
                            3'd7: matrix_in_values.vector3.p3 <= in_value_i;
                            default: ;
                        endcase
                        if (counter_pool == 3'd7) begin
                            counter_pool   <= '0;
                            counter_pixels <= counter_pixels + 1'b1;
                            px_ready       <= '1;
                        end else begin
                            counter_pool <= counter_pool + 1'b1;
                        end
                    end
                end

                default: ;
            endcase
        end
    end

    always_ff @(posedge clk_i) begin
        if (!nreset_i) begin
            out_reg  <= '0;
            px_rdy_o <= '0;
        end else begin
            px_rdy_o <= '0;
            if (px_ready) begin
                out_reg  <= pool_result;
                px_rdy_o <= '1;
            end
        end
    end

    assign out_px_o = out_reg;

endmodule