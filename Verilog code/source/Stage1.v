module Stage1 (
    input          		    i_clk,         
    input           	    i_rstn,      
    input					i_valid,
    input   		[ 7:0]  i_x,            
    input   		[ 1:0]  i_alpha,   	
    input   		[ 7:0]  i_inv_n,       	// 1/C
    
    output         		   	o_S1_done,      // Stage1 done 
    output  signed	[21:0]  o_Ex,          
    output  		[31:0]  o_Ex2,         
    output  signed	[ 8:0]  o_x_norm     	// To. input buffer, bit extension (signed)
);


    parameter	IDLE = 2'd0;
	parameter	BUSY = 2'd1;
	parameter	DONE = 2'd2;


	////////////// F/F ///////////////
	reg 		[1:0]  n_state,	c_state;
//    reg signed  [8:0]  n_x_norm, 	c_x_norm;  // bit extension (signed) 		
    reg 		[2:0]  n_cnt, 		c_cnt;
	

	///////////// wire /////////////
	wire				valid;
	wire signed [ 8:0]	temp_x_norm; 
	wire signed	[ 8:0] 	abs_x_norm;		// bit extension (signed)
	
	wire signed [21:0]	temp_Ex;
	wire		[31:0]	temp_Ex2;

	wire 				Ex_done;
	wire 				Ex2_done;


	// F/F
	always @(posedge i_clk, negedge i_rstn) 
    begin
        if (!i_rstn)
        begin
        	c_state	  	<= 2'd0;
//            c_x_norm 	<= 8'sd0;
            c_cnt 		<= 3'd0;
        end

        else 
        begin
        	c_state   	<= n_state;
//        	c_x_norm	<= n_x_norm;
        	c_cnt 		<= n_cnt;
        end
    end


	// n_state
	always @(*)
	begin
		n_state = c_state;
		case (c_state)
		IDLE : if (i_valid)			 	n_state = BUSY;
		BUSY : if (c_cnt == 3'd7)		n_state = DONE;
		DONE : if (o_S1_done)			n_state = IDLE;
		endcase
	end


	// n_cnt
	always @(*)
	begin
		n_cnt = c_cnt;
		case (c_state)
		IDLE :								n_cnt = 1'd0;
		//BUSY :	if (Ex_valid && Ex2_valid)	n_cnt = c_cnt + 1;
		BUSY :	if (i_valid)				n_cnt = c_cnt + 1;
		endcase
	end

/*
	// n_x_norm
	always @(*)
	begin
		n_x_norm = c_x_norm;
		case (c_state)
		IDLE : 			 		n_x_norm = 9'sd0;
		BUSY : 					n_x_norm = i_x - 128;
		endcase
	end
*/


    // x - zero point (128)
    assign temp_x_norm = (c_state == BUSY) ? i_x - 8'd128 : 9'sd0;	// signed -> Ex_Unit, input buffer	
	assign abs_x_norm = (c_state == BUSY) ? ((temp_x_norm > 0) ? temp_x_norm : -temp_x_norm) : 9'sd0;		// unsigned -> Ex2_Unit
//	assign abs_x_norm = (c_x_norm > 0 ) ? c_x_norm : -c_x_norm;
//    assign temp_x_norm = (i_valid) ? i_x - 8'd128 : 9'sd0;	// signed -> Ex_Unit, input buffer	
//	assign abs_x_norm = (i_valid) ? ((temp_x_norm > 0) ? temp_x_norm : -temp_x_norm) : 9'sd0;		// unsigned -> Ex2_Unit

	// valid
//	assign Ex_valid = (i_x || c_x_norm) ? 1 : 0;
//	assign Ex2_valid = (i_x || abs_x_norm) ? 1 : 0;


	// output
	//assign o_x_norm = c_x_norm;
	assign o_x_norm = temp_x_norm;
	assign o_Ex  = (c_state == DONE) ? temp_Ex  : 22'sd0;
	assign o_Ex2 = (c_state == DONE) ? temp_Ex2 : 32'd0;


    // Ex Unit 
    Ex_Unit u_Ex_Unit (
        .i_clk(i_clk),
        .i_rstn(i_rstn),
//        .i_valid(Ex_valid), 
        .i_valid(i_valid), 
//    	.i_x(c_x_norm),
		.i_x(temp_x_norm),
        .i_alpha(i_alpha),
        .i_inv_n(i_inv_n),
        .o_Ex_done(Ex_done),
        .o_Ex(temp_Ex)
    );



    // Ex2 Unit 
    Ex2_Unit u_Ex2_Unit (
        .i_clk(i_clk),
        .i_rstn(i_rstn),
//        .i_valid(Ex2_valid),
        .i_valid(i_valid), 
        .i_x(abs_x_norm),
        .i_alpha(i_alpha),
        .i_inv_n(i_inv_n),
        .o_Ex2_done(Ex2_done),
        .o_Ex2(temp_Ex2)
    );


    assign o_S1_done = (Ex_done && Ex2_done) ? 1 : 0;
endmodule
