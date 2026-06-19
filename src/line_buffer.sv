`ifdef COCOTB_SIM
  `include "../src/parameters.svh"
`else
  `include "parameters.svh"
`endif

// -----------------------------------------------------------------------------
// line_buffer.sv  —  Opción A: acumula 2 filas, entrega 3 píxeles en paralelo
//
// Parámetro:
//   IMG_WIDTH  — ancho de imagen en píxeles (se pasa al instanciar)
//   DATA_WIDTH — ancho de cada píxel en bits (default: BITS_Q4_6 = 11)
//
// Protocolo de entrada (igual que conv_control):
//   px_rdy_i = 1  →  in_px_i es válido este ciclo
//   Los píxeles llegan en orden raster: fila 0 de izquierda a derecha,
//   luego fila 1, etc.
//
// Salida:
//   out_row0_o  ←  píxel de la fila n-2 en la posición actual
//   out_row1_o  ←  píxel de la fila n-1
//   out_row2_o  ←  píxel de la fila n   (el que acaba de entrar)
//   px_rdy_o   = 1 cuando las 3 filas son válidas (a partir de la fila 2)
//
// Las 3 salidas se conectan a 3 entradas separadas de conv_control,
// que las acumula en su matrix_3x3 con su FSM existente.
//
// Implementación:
//   Dos SRAMs (arrays de registros) de IMG_WIDTH × DATA_WIDTH bits.
//   El puntero col_ptr recorre 0..IMG_WIDTH-1 y row_cnt cuenta filas.
//   row0_buf ← datos de hace 2 filas
//   row1_buf ← datos de la fila anterior
//   La fila actual se lee directamente desde in_px_i.
// -----------------------------------------------------------------------------

module line_buffer #(
    parameter int IMG_WIDTH  = 32,
    parameter int DATA_WIDTH = BITS_Q4_6   // 11 bits Q4.6
)(
    input  logic                    clk_i,
    input  logic                    nreset_i,

    input  logic                    px_rdy_i,
    input  logic [DATA_WIDTH-1:0]   in_px_i,

    output logic [DATA_WIDTH-1:0]   out_row0_o,   // fila n-2
    output logic [DATA_WIDTH-1:0]   out_row1_o,   // fila n-1
    output logic [DATA_WIDTH-1:0]   out_row2_o,   // fila n  (actual)
    output logic                    px_rdy_o
);

    // -------------------------------------------------------------------------
    // Buffers de línea: dos filas completas
    // -------------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] row0_buf [0:IMG_WIDTH-1];  // fila n-2
    logic [DATA_WIDTH-1:0] row1_buf [0:IMG_WIDTH-1];  // fila n-1

    // -------------------------------------------------------------------------
    // Contadores
    // -------------------------------------------------------------------------
    logic [$clog2(IMG_WIDTH)-1:0]  col_ptr;    // posición horizontal actual
    logic [MAX_RESOLUTION_BITS-1:0] row_cnt;   // fila actual (cuenta desde 0)

    // -------------------------------------------------------------------------
    // Escritura y desplazamiento de filas
    // -------------------------------------------------------------------------
    always_ff @(posedge clk_i) begin
        if (!nreset_i) begin
            col_ptr <= '0;
            row_cnt <= '0;
        end else if (px_rdy_i) begin
            // Guardar el píxel actual en row1_buf
            // (row1_buf siempre recibe la fila que acaba de completarse)
            // En la posición actual se escribe lo que era row1 en row0,
            // y el píxel entrante en row1.
            row0_buf[col_ptr] <= row1_buf[col_ptr];
            row1_buf[col_ptr] <= in_px_i;

            if (col_ptr == IMG_WIDTH - 1) begin
                col_ptr <= '0;
                row_cnt <= row_cnt + 1'b1;
            end else begin
                col_ptr <= col_ptr + 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Salidas combinacionales
    // -------------------------------------------------------------------------
    // out_row2 = píxel actual (directo desde la entrada)
    // out_row1 = lo que hay en row1_buf ANTES de ser sobreescrito
    //            → se lee con col_ptr actual (aún no actualizado)
    // out_row0 = lo que hay en row0_buf ANTES de ser sobreescrito

    assign out_row2_o = in_px_i;
    assign out_row1_o = row1_buf[col_ptr];
    assign out_row0_o = row0_buf[col_ptr];

    // px_rdy_o válido solo cuando ya hay al menos 2 filas almacenadas
    // (row_cnt >= 2) y hay un píxel entrante
    assign px_rdy_o = px_rdy_i && (row_cnt >= 2);

endmodule