//======================================================================
// Module: Imp_mult4b_lut
// Description: 4bit square LUT (16-entry ROM)
//              Used in Ex2_Unit
//======================================================================

module Imp_mult4b_lut (
	input		[3:0]	addr,
	output reg	[7:0]	data
);


	always@(*)
	begin
		case (addr)
			4'd0	: data = 8'd0;
			4'd1	: data = 8'd1;
			4'd2	: data = 8'd4;	
			4'd3	: data = 8'd9;
			4'd4	: data = 8'd16;
			4'd5	: data = 8'd25;
			4'd6	: data = 8'd36;
			4'd7	: data = 8'd49;
			4'd8	: data = 8'd64;
			4'd9	: data = 8'd81;
			4'd10	: data = 8'd100;
			4'd11	: data = 8'd121;
			4'd12	: data = 8'd144;
			4'd13	: data = 8'd169;
			4'd14	: data = 8'd196;
			4'd15	: data = 8'd225;
			default : data = 8'd0;
		endcase
	end

endmodule
