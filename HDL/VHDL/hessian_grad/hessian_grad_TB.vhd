LIBRARY IEEE;
PACKAGE hessian_constants IS
    CONSTANT HESSIAN_OUTPUT_Y : INTEGER := 90;
    CONSTANT HESSIAN_OUTPUT_X : INTEGER := 90;
END PACKAGE;

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
USE IEEE.STD_LOGIC_TEXTIO.ALL;
USE STD.TEXTIO.ALL;
USE work.hessian_constants.ALL;

ENTITY hessian_grad_TB IS
END hessian_grad_TB;

ARCHITECTURE Behavioral OF hessian_grad_TB IS
    SIGNAL clk : STD_LOGIC := '0';
    SIGNAL rst : STD_LOGIC := '1';
    CONSTANT clk_period : TIME := 10 ns;

    -- Control
    SIGNAL start_r, done_r : STD_LOGIC := '0';
    SIGNAL start_c, done_c : STD_LOGIC := '0';
    SIGNAL start_rr_cc, done_rr_cc : STD_LOGIC := '0';

    -- Memories
    CONSTANT RAM_SIZE : INTEGER := HESSIAN_OUTPUT_Y * HESSIAN_OUTPUT_X;
    TYPE ram_type IS ARRAY (0 TO RAM_SIZE - 1) OF signed(15 DOWNTO 0);
    SIGNAL ram_conv : ram_type := (OTHERS => (OTHERS => '0'));
    SIGNAL ram_gr : ram_type := (OTHERS => (OTHERS => '0'));
    SIGNAL ram_gc : ram_type := (OTHERS => (OTHERS => '0'));
    SIGNAL ram_rr_p_cc : ram_type := (OTHERS => (OTHERS => '0'));
    SIGNAL ram_rr_m_cc : ram_type := (OTHERS => (OTHERS => '0'));

    -- Interconnects
    SIGNAL gr_conv0_addr, gr_conv1_addr, gr_addr : STD_LOGIC_VECTOR(13 DOWNTO 0);
    SIGNAL gr_conv0_dout, gr_conv1_dout, gr_din : signed(15 DOWNTO 0);
    SIGNAL gr_we : STD_LOGIC;

    SIGNAL gc_conv0_addr, gc_conv1_addr, gc_addr : STD_LOGIC_VECTOR(13 DOWNTO 0);
    SIGNAL gc_conv0_dout, gc_conv1_dout, gc_din : signed(15 DOWNTO 0);
    SIGNAL gc_we : STD_LOGIC;

    SIGNAL rr_gr0_addr, rr_gr1_addr, rr_gc0_addr, rr_gc1_addr : STD_LOGIC_VECTOR(13 DOWNTO 0);
    SIGNAL rr_gr0_dout, rr_gr1_dout, rr_gc0_dout, rr_gc1_dout : signed(15 DOWNTO 0);
    SIGNAL rr_p_cc_addr, rr_m_cc_addr : STD_LOGIC_VECTOR(13 DOWNTO 0);
    SIGNAL rr_p_cc_din, rr_m_cc_din : signed(15 DOWNTO 0);
    SIGNAL rr_p_cc_we, rr_m_cc_we : STD_LOGIC;

    -- Components
    COMPONENT hessian_grad_r
        PORT (
            clk : IN STD_LOGIC;
            rst : IN STD_LOGIC;
            start : IN STD_LOGIC;
            done : OUT STD_LOGIC;
            conv0_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            conv0_dout : IN signed(15 DOWNTO 0);
            conv1_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            conv1_dout : IN signed(15 DOWNTO 0);
            gr_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            gr_din : OUT signed(15 DOWNTO 0);
            gr_we : OUT STD_LOGIC);
    END COMPONENT;

    COMPONENT hessian_grad_c
        PORT (
            clk : IN STD_LOGIC;
            rst : IN STD_LOGIC;
            start : IN STD_LOGIC;
            done : OUT STD_LOGIC;
            conv0_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            conv0_dout : IN signed(15 DOWNTO 0);
            conv1_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            conv1_dout : IN signed(15 DOWNTO 0);
            gc_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            gc_din : OUT signed(15 DOWNTO 0);
            gc_we : OUT STD_LOGIC);
    END COMPONENT;

    COMPONENT hessian_grad_rr_cc
        PORT (
            clk : IN STD_LOGIC;
            rst : IN STD_LOGIC;
            start : IN STD_LOGIC;
            done : OUT STD_LOGIC;
            gr0_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            gr0_dout : IN signed(15 DOWNTO 0);
            gr1_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            gr1_dout : IN signed(15 DOWNTO 0);
            gc0_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            gc0_dout : IN signed(15 DOWNTO 0);
            gc1_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            gc1_dout : IN signed(15 DOWNTO 0);
            rr_p_cc_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            rr_p_cc_din : OUT signed(15 DOWNTO 0);
            rr_p_cc_we : OUT STD_LOGIC;
            rr_m_cc_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            rr_m_cc_din : OUT signed(15 DOWNTO 0);
            rr_m_cc_we : OUT STD_LOGIC);
    END COMPONENT;

BEGIN
    -- Clock
    clk_process : PROCESS BEGIN
        clk <= '0';
        WAIT FOR clk_period/2;
        clk <= '1';
        WAIT FOR clk_period/2;
    END PROCESS;

    -- RAM Models (Synchronous)
    PROCESS (clk) BEGIN
        IF rising_edge(clk) THEN
            -- grad_r reads
            gr_conv0_dout <= ram_conv(to_integer(unsigned(gr_conv0_addr)));
            gr_conv1_dout <= ram_conv(to_integer(unsigned(gr_conv1_addr)));
            -- grad_c reads
            gc_conv0_dout <= ram_conv(to_integer(unsigned(gc_conv0_addr)));
            gc_conv1_dout <= ram_conv(to_integer(unsigned(gc_conv1_addr)));
            -- grad_rr_cc reads
            rr_gr0_dout <= ram_gr(to_integer(unsigned(rr_gr0_addr)));
            rr_gr1_dout <= ram_gr(to_integer(unsigned(rr_gr1_addr)));
            rr_gc0_dout <= ram_gc(to_integer(unsigned(rr_gc0_addr)));
            rr_gc1_dout <= ram_gc(to_integer(unsigned(rr_gc1_addr)));
            -- writes
            IF gr_we = '1' THEN
                ram_gr(to_integer(unsigned(gr_addr))) <= gr_din;
            END IF;
            IF gc_we = '1' THEN
                ram_gc(to_integer(unsigned(gc_addr))) <= gc_din;
            END IF;
            IF rr_p_cc_we = '1' THEN
                ram_rr_p_cc(to_integer(unsigned(rr_p_cc_addr))) <= rr_p_cc_din;
            END IF;
            IF rr_m_cc_we = '1' THEN
                ram_rr_m_cc(to_integer(unsigned(rr_m_cc_addr))) <= rr_m_cc_din;
            END IF;
        END IF;
    END PROCESS;

    -- Instantiation
    uut_grad_r : hessian_grad_r PORT MAP(clk, rst, start_r, done_r, gr_conv0_addr, gr_conv0_dout, gr_conv1_addr, gr_conv1_dout, gr_addr, gr_din, gr_we);
    uut_grad_c : hessian_grad_c PORT MAP(clk, rst, start_c, done_c, gc_conv0_addr, gc_conv0_dout, gc_conv1_addr, gc_conv1_dout, gc_addr, gc_din, gc_we);
    uut_grad_rr_cc : hessian_grad_rr_cc PORT MAP(clk, rst, start_rr_cc, done_rr_cc, rr_gr0_addr, rr_gr0_dout, rr_gr1_addr, rr_gr1_dout, rr_gc0_addr, rr_gc0_dout, rr_gc1_addr, rr_gc1_dout, rr_p_cc_addr, rr_p_cc_din, rr_p_cc_we, rr_m_cc_addr, rr_m_cc_din, rr_m_cc_we);

    -- Stimulus
    stim_proc : PROCESS
        FILE input_file : text;
        VARIABLE input_line : line;
        VARIABLE input_val : INTEGER;
        VARIABLE i : INTEGER;
        FILE output_file_gr, output_file_gc, output_file_rr_p, output_file_rr_m : text;
        VARIABLE output_line : line;
    BEGIN
        REPORT "Loading input data...";
        file_open(input_file, "input.txt", read_mode);
        i := 0;
        -- ROBUST READ LOOP
        WHILE NOT endfile(input_file) AND i < RAM_SIZE LOOP
            readline(input_file, input_line);
            IF input_line'length > 0 AND input_line(1) /= ' ' THEN
                read(input_line, input_val);
                ram_conv(i) <= to_signed(input_val, 16);
                i := i + 1;
            END IF;
        END LOOP;
        file_close(input_file);
        REPORT "Loaded " & INTEGER'image(i) & " values.";

        -- Run Simulation
        rst <= '1';
        WAIT FOR clk_period * 5;
        rst <= '0';
        WAIT FOR clk_period * 2;

        start_r <= '1';
        WAIT FOR clk_period;
        start_r <= '0';
        WAIT UNTIL done_r = '1';
        WAIT FOR clk_period * 5;

        start_c <= '1';
        WAIT FOR clk_period;
        start_c <= '0';
        WAIT UNTIL done_c = '1';
        WAIT FOR clk_period * 5;

        start_rr_cc <= '1';
        WAIT FOR clk_period;
        start_rr_cc <= '0';
        WAIT UNTIL done_rr_cc = '1';
        WAIT FOR clk_period * 5;

        -- Dump Results
        REPORT "Writing output files...";
        file_open(output_file_gr, "vhdl_gr_out.txt", write_mode);
        FOR j IN 0 TO RAM_SIZE - 1 LOOP write(output_line, to_integer(ram_gr(j)));
            writeline(output_file_gr, output_line);
        END LOOP;
        file_close(output_file_gr);

        file_open(output_file_gc, "vhdl_gc_out.txt", write_mode);
        FOR j IN 0 TO RAM_SIZE - 1 LOOP write(output_line, to_integer(ram_gc(j)));
            writeline(output_file_gc, output_line);
        END LOOP;
        file_close(output_file_gc);

        file_open(output_file_rr_p, "vhdl_rr_p_cc_out.txt", write_mode);
        FOR j IN 0 TO RAM_SIZE - 1 LOOP write(output_line, to_integer(ram_rr_p_cc(j)));
            writeline(output_file_rr_p, output_line);
        END LOOP;
        file_close(output_file_rr_p);

        file_open(output_file_rr_m, "vhdl_rr_m_cc_out.txt", write_mode);
        FOR j IN 0 TO RAM_SIZE - 1 LOOP write(output_line, to_integer(ram_rr_m_cc(j)));
            writeline(output_file_rr_m, output_line);
        END LOOP;
        file_close(output_file_rr_m);

        REPORT "Simulation Finished Successfully.";
        ASSERT false REPORT "End of simulation" SEVERITY failure;
    END PROCESS;
END Behavioral;