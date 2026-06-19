module multiplier (
input [1:0] A,
input [1:0] B,
output [3:0] P);

// Generación de productos parciales
 wire pp0 = ((A[0]) & (B[0]));
 wire pp1 = ((1) & (~B[0]));
 wire pp2 = ((~A[0]) & (~B[1]));
 wire pp3 = ((~A[1]) & (~B[1]));
 wire pp4 = ((A[1]) & (B[0]));
 wire pp5 = ((A[1]) & (B[1]));
 wire pp6 = ((A[0]) & (B[1]));

 // Suma de productos parciales
wire [1:0] columna4 = pp2 + pp2;
wire [1:0] columna3 = pp5 + pp3;
wire [1:0] columna2 = pp0 + pp4;
wire [1:0] columna1 = pp6 + pp1;
assign P = ({2'b00, columna4} << 3) + 
           ({2'b00, columna3} << 2) + 
           ({2'b00, columna2} << 1) + 
           ({2'b00, columna1} << 0);


endmodule
