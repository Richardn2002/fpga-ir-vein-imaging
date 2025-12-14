LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
LIBRARY UNISIM;
USE UNISIM.vcomponents.ALL;

ENTITY cmt IS
	PORT (
		sysclk : IN STD_LOGIC; -- 12 MHz sysclk
		cam_xclk : OUT STD_LOGIC; -- 24 MHz clock input to camera
		cam_ctrl_clk : OUT STD_LOGIC; -- 12 MHz clock input to camera control
		vga_clk : OUT STD_LOGIC; -- 25.153 MHz clock input to VGA module
		core_clk : OUT STD_LOGIC -- 36 MHz core clock
	);
END cmt;

ARCHITECTURE arch OF cmt IS
	SIGNAL clk_fb : STD_LOGIC;
BEGIN
	-- This module generates 720 MHz VCO from 12 MHz input clock,
	-- then divides into all clocks the whole system uses
	--
	-- Ffbo = Fclkin / DIVCLK_DIVIDE = Fvco / CLKFBOUT_MULT_F
	--
	-- So:
	-- Fvco = Fclkin * CLKFBOUT_MULT_F / DIVCLK_DIVIDE
	--
	-- CLKFBOUT_MULT_F = 60.0
	-- DIVCLK_DIVIDE = 1.0
	--
	cmt : MMCME2_BASE GENERIC MAP(
		-- Jitter programming (OPTIMIZED, HIGH, LOW)
		BANDWIDTH => "OPTIMIZED",
		-- Multiply value for all CLKOUT (2.000-64.000).
		-- VCO 720 MHz -> CLKFBOUT -> CLKFBIN 12 MHz
		CLKFBOUT_MULT_F => 60.0,
		-- Phase offset in degrees of CLKFB (-360.000-360.000).
		CLKFBOUT_PHASE => 0.0,
		-- Input clock period in ns to ps resolution (i.e. 33.333 is 30 MHz).
		CLKIN1_PERIOD => 83.333,
		-- Divide amount for each CLKOUT (1-128)
		CLKOUT1_DIVIDE => 30, -- 720 / 30 = 24 to camera
		CLKOUT2_DIVIDE => 60, -- 720 / 60 = 12 to camera control
		CLKOUT3_DIVIDE => 20, -- 720 / 20 = 36 to core
		CLKOUT4_DIVIDE => 1,
		CLKOUT5_DIVIDE => 1,
		CLKOUT6_DIVIDE => 1,
		-- Divide amount for CLKOUT0 (1.000-128.000):
		CLKOUT0_DIVIDE_F => 28.625, -- 720 / 28.625 = 25.153 MHz pixel clock to VGA
		-- Duty cycle for each CLKOUT (0.01-0.99):
		CLKOUT0_DUTY_CYCLE => 0.5,
		CLKOUT1_DUTY_CYCLE => 0.5,
		CLKOUT2_DUTY_CYCLE => 0.5,
		CLKOUT3_DUTY_CYCLE => 0.5,
		CLKOUT4_DUTY_CYCLE => 0.5,
		CLKOUT5_DUTY_CYCLE => 0.5,
		CLKOUT6_DUTY_CYCLE => 0.5,
		-- Phase offset for each CLKOUT (-360.000-360.000):
		CLKOUT0_PHASE => 0.0,
		CLKOUT1_PHASE => 0.0,
		CLKOUT2_PHASE => 0.0,
		CLKOUT3_PHASE => 0.0,
		CLKOUT4_PHASE => 0.0,
		CLKOUT5_PHASE => 0.0,
		CLKOUT6_PHASE => 0.0,
		-- Cascade CLKOUT4 counter with CLKOUT6 (FALSE, TRUE)
		CLKOUT4_CASCADE => FALSE,
		-- Master division value (1-106)
		-- CLKIN1 12 MHz -> CLKFBIN 12 MHz
		DIVCLK_DIVIDE => 1,
		-- Reference input jitter in UI (0.000-0.999).
		REF_JITTER1 => 0.0,
		-- Delays DONE until MMCM is locked (FALSE, TRUE)
		STARTUP_WAIT => FALSE
		) PORT MAP (
		-- User Configurable Clock Outputs:
		CLKOUT0 => vga_clk, -- 1-bit output: CLKOUT0
		CLKOUT0B => OPEN, -- 1-bit output: Inverted CLKOUT0
		CLKOUT1 => cam_xclk, -- 1-bit output: CLKOUT1
		CLKOUT1B => OPEN, -- 1-bit output: Inverted CLKOUT1
		CLKOUT2 => cam_ctrl_clk, -- 1-bit output: CLKOUT2
		CLKOUT2B => OPEN, -- 1-bit output: Inverted CLKOUT2
		CLKOUT3 => core_clk, -- 1-bit output: CLKOUT3
		CLKOUT3B => OPEN, -- 1-bit output: Inverted CLKOUT3
		CLKOUT4 => OPEN, -- 1-bit output: CLKOUT4
		CLKOUT5 => OPEN, -- 1-bit output: CLKOUT5
		CLKOUT6 => OPEN, -- 1-bit output: CLKOUT6
		-- Clock Feedback Output Ports:
		CLKFBOUT => clk_fb, -- 1-bit output: Feedback clock
		CLKFBOUTB => OPEN, -- 1-bit output: Inverted CLKFBOUT
		-- MMCM Status Ports:
		LOCKED => OPEN, -- 1-bit output: LOCK
		-- Clock Input:
		CLKIN1 => sysclk, -- 1-bit input: Clock
		-- MMCM Control Ports:
		PWRDWN => '0', -- 1-bit input: Power-down
		RST => '0', -- 1-bit input: Reset
		-- Clock Feedback Input Port:
		CLKFBIN => clk_fb -- 1-bit input: Feedback clock
	);
END arch;