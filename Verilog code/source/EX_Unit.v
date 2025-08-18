module Ex_Unit(
    input           		i_clk,        
    input           		i_rstn,      
    input           		i_valid,            
    input  signed	[8:0]   i_x,     
    input   		[1:0]   i_alpha,   // PTF scailing �� 
    input   		[7:0]   i_inv_n,   // 1/C, Q0.8 -> LUT


    output          		o_Ex_done,
    output signed 	[21:0]  o_Ex       // Ex ���
);
	
	parameter	IDLE = 2'd0;
	parameter	ACC	 = 2'd1;
	parameter	DONE = 2'd2;

	/////////////// F/F /////////////////
	reg 		[ 1:0]	n_state,   	 c_state;
    reg signed 	[14:0] 	n_acc, 	  	 c_acc;   		// 11 bits + log2(N) = 11 + 3 + 1(sign) -> 15 bits
    reg 		[ 2:0] 	n_cnt_acc, 	 c_cnt_acc;
//	reg signed	[10:0] 	n_shifted_x, c_shifted_x;


	/////////////// reg /////////////////
//	reg [23:0] mult_ex;       // acc(12bits) * i_inv_n(12bits) -> 24 bits 
//    reg [15:0] temp_ex;



	/////////////// wire ////////////////
    wire signed	[10:0] shifted_x;			// 9bits << 2 -> 11bits
    wire signed	[10:0] padded_inv_n;
	wire signed	[29:0] mult_ex;				// acc(15bits) * inv_n(15bits) -> 30bits 
	wire signed	[21:0] temp_ex;				// 30bits >> 8 -> 22bits




    // F/F
    always @(posedge i_clk or negedge i_rstn) 
    begin
        if (!i_rstn)
        begin
        	c_state	  	<= 2'd0;
//        	c_shifted_x	<= 11'sd0;
            c_acc 	  	<= 15'sd0;
            c_cnt_acc 	<= 3'd0;
        end

        else 
        begin
        	c_state   	<= n_state;
//        	c_shifted_x <= n_shifted_x;
        	c_acc	  	<= n_acc;
        	c_cnt_acc 	<= n_cnt_acc;
        end
    end


	// n_state
	always @(*)
	begin
		n_state = c_state;
		case (c_state)
		IDLE : if (i_valid)			 	n_state = ACC;
		ACC  : if (c_cnt_acc == 3'd7)	n_state = DONE;
		DONE : if (o_Ex_done)			n_state = IDLE;
		endcase
	end


	// n_cnt_acc
	always @(*)
	begin
		n_cnt_acc = c_cnt_acc;
		case (c_state)
		IDLE :							n_cnt_acc = 1'd0;
		ACC  :	if (i_valid)			n_cnt_acc = c_cnt_acc + 1;
		endcase
	end


	// n_acc
	always @(*)
	begin
		n_acc = c_acc;
		case (c_state)
		IDLE : 							n_acc = 15'sd0;
		ACC  :							n_acc = c_acc + shifted_x;
		endcase
	end

	

    // PTF scaling
    assign shifted_x = i_x << i_alpha;
	/*
	always @(*)
	begin
		n_shifted_x = c_shifted_x;
		case (c_state)
		IDLE :	if (i_valid)	n_shifted_x = i_x << i_alpha;
		ACC	 : 					n_shifted_x = i_x << i_alpha;
		DONE : 					n_shifted_x = 11'sd0;
		endcase
	end
	*/

	assign padded_inv_n = {7'b0, i_inv_n}; // 


	// Accumulator * inv_n (1/C) 
	/*
	always @(*)
	begin
		mult_ex = 24'd0;
		temp_ex = 16'd0;
		if (c_state == 2'd2)
		begin
			mult_ex = c_acc * padded_inv_n;
			temp_ex = mult_ex >> 8;
			o_Ex	= temp_ex[15:8];
		end
	end
    */

	assign mult_ex = (c_state == DONE) ? c_acc * padded_inv_n : 30'sd0;
	assign temp_ex = mult_ex >>> 8; // arithmatic right shift -> synthesis ?

	//assign o_Ex = (c_state == DONE) ? temp_ex[7:0] : 8'sd0;

	assign o_Ex = (c_state == DONE) ? temp_ex : 22'sd0;
	assign o_Ex_done = (c_state == DONE);


endmodule
