LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
USE std.textio.ALL;

USE work.constants;

ENTITY hessian_output_TB IS
END ENTITY;

ARCHITECTURE sim OF hessian_output_TB IS

    CONSTANT CLK_PERIOD : TIME := 10 ns;

    CONSTANT N : INTEGER := constants.HESSIAN_OUTPUT_X * constants.HESSIAN_OUTPUT_Y;

    SIGNAL clk : STD_LOGIC := '0';
    SIGNAL trg : STD_LOGIC := '0';
    SIGNAL rdy : STD_LOGIC;

    SIGNAL rrp_addr : unsigned(15 DOWNTO 0);
    SIGNAL rrm_addr : unsigned(15 DOWNTO 0);
    SIGNAL rc_addr : unsigned(15 DOWNTO 0);

    SIGNAL rrp_dout : signed(15 DOWNTO 0);
    SIGNAL rrm_dout : signed(15 DOWNTO 0);
    SIGNAL rc_dout : signed(15 DOWNTO 0);

    SIGNAL out_addr : unsigned(15 DOWNTO 0);
    SIGNAL out_din : signed(15 DOWNTO 0);
    SIGNAL out_we : STD_LOGIC;

    TYPE mem_s16_t IS ARRAY (0 TO N - 1) OF signed(15 DOWNTO 0);
    SIGNAL mem_rrp : mem_s16_t := (OTHERS => (OTHERS => '0'));
    SIGNAL mem_rrm : mem_s16_t := (OTHERS => (OTHERS => '0'));
    SIGNAL mem_rc : mem_s16_t := (OTHERS => (OTHERS => '0'));
    SIGNAL mem_out : mem_s16_t := (OTHERS => (OTHERS => '0'));

    SIGNAL rrp_addr_q : unsigned(15 DOWNTO 0) := (OTHERS => '0');
    SIGNAL rrm_addr_q : unsigned(15 DOWNTO 0) := (OTHERS => '0');
    SIGNAL rc_addr_q : unsigned(15 DOWNTO 0) := (OTHERS => '0');

    FILE f_rrp : text OPEN read_mode IS "/raid/fall2025/plal4/proj_out/proj_out.srcs/sim_1/new/stage07_rr_p_cc.txt";
    FILE f_rrm : text OPEN read_mode IS "/raid/fall2025/plal4/proj_out/proj_out.srcs/sim_1/new/stage08_rr_m_cc.txt";
    FILE f_rc : text OPEN read_mode IS "/raid/fall2025/plal4/proj_out/proj_out.srcs/sim_1/new/stage09_rc.txt";
    FILE f_out : text OPEN write_mode IS "/raid/fall2025/plal4/proj_out/proj_out.srcs/sim_1/new/vhdl_stage10_hessian_output.txt";

BEGIN

    clk <= NOT clk AFTER CLK_PERIOD/2;

    dut : ENTITY work.hessian_output
        PORT MAP(
            clk => clk,
            trg => trg,
            rdy => rdy,

            rrp_addr => STD_LOGIC_VECTOR(rrp_addr),
            rrp_dout => STD_LOGIC_VECTOR(rrp_dout),

            rrm_addr => STD_LOGIC_VECTOR(rrm_addr),
            rrm_dout => STD_LOGIC_VECTOR(rrm_dout),

            rc_addr => STD_LOGIC_VECTOR(rc_addr),
            rc_dout => STD_LOGIC_VECTOR(rc_dout),

            out_addr => STD_LOGIC_VECTOR(out_addr),
            out_din => STD_LOGIC_VECTOR(out_din),
            out_we => out_we
        );

    load_proc : PROCESS
        VARIABLE l : line;
        VARIABLE v : INTEGER;
        VARIABLE i : INTEGER := 0;
    BEGIN

        i := 0;
        WHILE (NOT endfile(f_rrp)) AND (i < N) LOOP
            readline(f_rrp, l);
            read(l, v);
            mem_rrp(i) <= to_signed(v, 16);
            i := i + 1;
        END LOOP;

        i := 0;
        WHILE (NOT endfile(f_rrm)) AND (i < N) LOOP
            readline(f_rrm, l);
            read(l, v);
            mem_rrm(i) <= to_signed(v, 16);
            i := i + 1;
        END LOOP;

        i := 0;
        WHILE (NOT endfile(f_rc)) AND (i < N) LOOP
            readline(f_rc, l);
            read(l, v);
            mem_rc(i) <= to_signed(v, 16);
            i := i + 1;
        END LOOP;

        WAIT;
    END PROCESS;

    bram_read_proc : PROCESS (clk)
        VARIABLE a : INTEGER;
    BEGIN
        IF rising_edge(clk) THEN

            rrp_addr_q <= rrp_addr;
            rrm_addr_q <= rrm_addr;
            rc_addr_q <= rc_addr;

            a := to_integer(rrp_addr_q);
            IF a >= 0 AND a < N THEN
                rrp_dout <= mem_rrp(a);
            ELSE
                rrp_dout <= (OTHERS => '0');
            END IF;

            a := to_integer(rrm_addr_q);
            IF a >= 0 AND a < N THEN
                rrm_dout <= mem_rrm(a);
            ELSE
                rrm_dout <= (OTHERS => '0');
            END IF;

            a := to_integer(rc_addr_q);
            IF a >= 0 AND a < N THEN
                rc_dout <= mem_rc(a);
            ELSE
                rc_dout <= (OTHERS => '0');
            END IF;
        END IF;
    END PROCESS;

    -- BRAM synchronous WRITE model + dump file
    bram_write_proc : PROCESS (clk)
        VARIABLE a : INTEGER;
        VARIABLE l : line;
        VARIABLE write_count : INTEGER := 0;
    BEGIN
        IF rising_edge(clk) THEN
            IF out_we = '1' THEN
                a := to_integer(out_addr);
                IF a >= 0 AND a < N THEN
                    mem_out(a) <= out_din;
                END IF;

                write(l, to_integer(out_din));
                writeline(f_out, l);

                write_count := write_count + 1;
                IF write_count = N THEN
                    REPORT "Wrote all outputs to vhdl_stage10_hessian_output.txt";
                END IF;
            END IF;
        END IF;
    END PROCESS;

    stim_proc : PROCESS
    BEGIN
        WAIT FOR CLK_PERIOD;

        trg <= '1';
        WAIT FOR CLK_PERIOD;
        trg <= '0';

        WAIT UNTIL rdy = '1';
        REPORT "DUT rdy=1 (final write completed). Simulation done.";

        WAIT;
    END PROCESS;

END ARCHITECTURE;