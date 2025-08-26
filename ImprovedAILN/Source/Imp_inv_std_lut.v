module Imp_inv_std_lut #(
    parameter     FILENAME = "inv_std.mem"
)(
    input  wire [7:0]   addr,   // std_u8
    output reg  [13:0]  data
);
    (* rom_style="block" *) reg [13:0] rom [0:255];
    initial $readmemh(FILENAME, rom);
    always @(*)
    begin 
        data = rom[(addr==8'd0) ? 8'd1 : addr]; // 0 보호
    end

endmodule
