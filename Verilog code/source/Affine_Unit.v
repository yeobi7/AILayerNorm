/*
module Affine_Unit(
    input         				i_clk,
    input         				i_rstn,
    input         				i_valid,
    input	signed		[ 8:0] 	i_x_norm,    
    input   			[ 1:0] 	i_alpha,  
    input   signed		[21:0] 	i_mean,      
    input   			[ 7:0] 	i_std,    
    input   			[ 7:0] 	i_gamma,     
    input   			[ 7:0] 	i_beta,      

 	output				  		o_Affine_done,	// Affine Unit done
    output	signed		[7:0] 	o_Norm			// sign bit + 8bits ?? 
    //output  [15:0] o_y
);

    


    /////////////// F/F /////////////////
//	reg 		  		n_state,  	c_state;
	reg			[ 2:0]	n_cnt,		c_cnt;
	reg					n_o_done,	c_o_done;
//    reg signed 	[ 7:0]  n_o_Norm,  c_o_Norm;

	reg signed	[10:0] 	shifted_x;		// i_x_norm (9bits) << alpha(2) -> 11bits 
	reg signed	[11:0] 	temp_numerator;		// 11bits - 9bits -> 12bits(overflow)
	reg signed 	[23:0]	numerator;
	reg signed	[15:0] 	temp_Norm; 		// overflow -> 17bits

	
	/////////////// reg /////////////////

	/////////////// wire ////////////////
//    wire signed	[10:0] shifted_x;		// i_x_norm (9bits) << alpha(2) -> 11bits 
//	wire signed	[11:0] temp_numerator;		// 11bits - 9bits -> 12bits(overflow)
//	wire signed [23:0]	numerator;
//	wire 		[15:0] temp_denom;
//	wire 		[ 7:0] denominator;	// sigma; affined_std

//	wire signed	[15:0] temp_Norm; 		// overflow -> 17bits



    // F/F
    always @(posedge i_clk or negedge i_rstn) 
    begin
        if (!i_rstn)
        begin
//        	c_state		<= 1'd0;
        	c_cnt		<= 3'd0;
        	c_o_done	<= 1'd0;
//            c_o_Norm <= 16'sd0;
        end

        else 
        begin
//        	c_state  	<= n_state;
        	c_cnt		<= n_cnt;
        	c_o_done	<= n_o_done;
//        	c_o_Norm <= n_o_Norm;
        end
    end


/*
	// n_state
	always @(*)
	begin
		n_state = c_state;
		case (c_state)
		1'b0 : if (i_valid)					n_state = 1'd1;	
		1'b1 : if (o_Affine_done)			n_state = 1'b0;
		endcase
	end
*/

/*
	// n_cnt -> FSM
	always @(*)
	begin
		n_cnt = c_cnt;
		case (c_state)
		1'b0 : 								n_cnt = 3'd0;	
		1'b1 : 								n_cnt = c_cnt + 1;
		endcase
	end
*/


/*************************
// n_cnt -> Not FSM
	always @(*)
	begin
		n_cnt = c_cnt;	
		if (o_Affine_done)						n_cnt = 3'd0;	
		else if (i_valid)						n_cnt = c_cnt + 1;

		
	end


	// n_o_done
	always @(*)
	begin
		n_o_done = c_o_done;
		if (c_cnt == 3'd7)						n_o_done = 1'd1;	
		else 									n_o_done = 1'd0;
	end
*******************************/
	
    // PTF scailing
//    assign shifted_x = i_x_norm << i_alpha;

/*		
    always @(*)
    begin
    	shifted_x		= 11'sd0;
		temp_numerator 	= 12'sd0;
		numerator		= 24'sd0;
		temp_Norm		= 16'sd0; 
		case (c_state)			
		1'd1 :	begin
					shifted_x		= i_x_norm << i_alpha;
					temp_numerator 	= shifted_x - i_mean;
					numerator		= temp_numerator * {4'b0, i_gamma};
					temp_Norm		= (numerator * i_std) >> 8;	
				end
		endcase
    end
 */

/***************************************
	// Not FSM
    always @(*)
    begin
    	shifted_x		= 11'sd0;
		temp_numerator 	= 12'sd0;
		numerator		= 24'sd0;
		temp_Norm		= 16'sd0;
		if (o_Affine_done)	
			begin
				shifted_x		= 11'sd0;
				temp_numerator 	= 12'sd0;
				numerator		= 24'sd0;
				temp_Norm		= 16'sd0;
			end

		else if (i_valid)		
			begin
				shifted_x		= i_x_norm << i_alpha;
				temp_numerator 	= shifted_x - i_mean;
//				numerator		= temp_numerator * {4'b0, i_gamma};
				numerator 		= temp_numerator * $signed({4'b0, i_gamma});			
				temp_Norm		= (numerator * i_std) >>> 8;	
			end
    end

*************************************/

    // shifted_x - mean -> (X - u) * gamma
//    assign temp_numerator = (shifted_x - i_mean); 			// 11bits - 9bits -> 12bits(overflow)
//	assign numerator = temp_numerator * {4'b0, i_gamma}; 	// 24bits 

	// {(X - u) * gamma} * 1/C
//	assign temp_Norm = (numerator * i_std) >> 8;



/*
    // gamma * std -> sigma
//    assign temp_denom = i_gamma * i_std;
//    assign denominator = temp_denom >> 8;	// Q0.8 -> INT8
//	assign denominator = (i_gamma * i_std) >> 8;

    // numerator * denominator + beta -> Y = (X - u) / sigma   
    //assign temp_Norm = (numerator * {4'b0, denominator}) + {16'b0, i_beta};		// 12bits * 12bits -> 24bits		

	// n_o_Norm = sign bit + {((X - u) * gamma) / sigma} + beta
*/


/*
	always @(*)
	begin
		n_o_Norm = c_o_Norm;
		if (i_x_norm)		// ************************* case (c_state) *******************
			n_o_Norm = o_Norm;
		//endcase
	end
*/

	
/***************************
    assign o_Affine_done = c_o_done; 
//	assign o_Norm = (i_valid) ? temp_Norm[7:0] + i_beta : 8'sd0;
	assign o_Norm = (i_valid) ? $signed(temp_Norm[7:0]) + i_beta : 8'sd0;

endmodule

******************************/



// 수정 대상 1: Affine_Unit 파이프라이닝
// 기존에는 조합 로직이 4줄 한 싸이클에 처리되어 hold violation 유발
// 해결: 입력 데이터에 대한 연산을 두 단계로 나눠 파이프라인 설계 + 연산 단순화 (shift, sub)

module Affine_Unit #(parameter WIDTH = 8)(
    input                 i_clk,
    input                 i_rstn,
    input                 i_valid,
    input signed  [ 8:0]  i_x_norm,
    input        [ 1:0]   i_alpha,
    input signed [21:0]   i_mean,
    input        [ 7:0]   i_std,
    input        [ 7:0]   i_gamma,
    input        [ 7:0]   i_beta,

    output reg            o_Affine_done,
    output signed [ 7:0]  o_Norm
);

    // 파이프라인 레지스터 선언
    reg signed [10:0] r_shifted_x;
    reg signed [11:0] r_temp_numerator;
    reg signed [23:0] r_numerator;
    reg signed [15:0] r_temp_Norm;

    reg [1:0] c_state, n_state;
    localparam IDLE = 2'd0, STAGE1 = 2'd1, STAGE2 = 2'd2;

    // 상태 레지스터
    always @(posedge i_clk or negedge i_rstn) begin
        if (!i_rstn) begin
            c_state          <= IDLE;
            r_shifted_x      <= 11'sd0;
            r_temp_numerator <= 12'sd0;
            r_numerator      <= 24'sd0;
            r_temp_Norm      <= 16'sd0;
            o_Affine_done    <= 1'b0;
        end else begin
            c_state          <= n_state;
            r_shifted_x      <= (i_valid && c_state == IDLE) ? (i_x_norm <<< i_alpha) : 11'sd0;
            r_temp_numerator <= (i_valid && c_state == IDLE) ? ((i_x_norm <<< i_alpha) - i_mean) : 12'sd0;
            r_numerator      <= (c_state == STAGE1) ? (r_temp_numerator * $signed({4'b0, i_gamma})) : 24'sd0;
            r_temp_Norm      <= (c_state == STAGE1) ? ((r_numerator * i_std) >>> 8) : 16'sd0;
            o_Affine_done    <= (c_state == STAGE2);
        end
    end

    // 상태 전이 로직
    always @(*) begin
        case (c_state)
            IDLE:    n_state = (i_valid) ? STAGE1 : IDLE;
            STAGE1:  n_state = STAGE2;
            STAGE2:  n_state = IDLE;
            default: n_state = IDLE;
        endcase
    end

    // 출력 결과
    assign o_Norm = (c_state == STAGE2) ? $signed(r_temp_Norm[7:0]) + i_beta : 8'sd0;

endmodule


