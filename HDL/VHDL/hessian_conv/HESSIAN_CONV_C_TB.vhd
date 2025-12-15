LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;
USE STD.textio.ALL;
USE IEEE.std_logic_textio.ALL;

ENTITY tb_HESSIAN_conv_c IS
    GENERIC (
        INPUT_FILE  : STRING := "C:\Users\cours\IR_IMAGING\HDL_TB\stage03_conv_r.txt";
        OUTPUT_FILE : STRING := "C:\Users\cours\IR_IMAGING\PP_hessianCONVC_output_90x90.txt"
    );
END tb_HESSIAN_conv_c;

ARCHITECTURE arch OF tb_HESSIAN_conv_c IS

    COMPONENT HESSIAN_conv_c
        PORT (
            clk           : IN  STD_LOGIC;
            trg           : IN  STD_LOGIC;
            rdy           : OUT STD_LOGIC;

            conv_in_en    : OUT STD_LOGIC;
            conv_in_addr  : OUT STD_LOGIC_VECTOR(12 DOWNTO 0);
            conv_in_d     : IN  STD_LOGIC_VECTOR(15 DOWNTO 0);

            conv_out_en   : OUT STD_LOGIC;
            conv_out_addr : OUT STD_LOGIC_VECTOR(12 DOWNTO 0);
            conv_out_d    : OUT STD_LOGIC_VECTOR(15 DOWNTO 0)
        );
    END COMPONENT;


    SIGNAL clk : STD_LOGIC := '0';
    CONSTANT clk_period : TIME := 10 ns;

    SIGNAL trg           : STD_LOGIC := '0';
    SIGNAL rdy           : STD_LOGIC;

    SIGNAL conv_in_en    : STD_LOGIC;
    SIGNAL conv_in_addr  : STD_LOGIC_VECTOR(12 DOWNTO 0);
    SIGNAL conv_in_d     : STD_LOGIC_VECTOR(15 DOWNTO 0);

    SIGNAL conv_out_en   : STD_LOGIC;
    SIGNAL conv_out_addr : STD_LOGIC_VECTOR(12 DOWNTO 0);
    SIGNAL conv_out_d    : STD_LOGIC_VECTOR(15 DOWNTO 0);


    TYPE mem_t IS ARRAY (0 TO 8100-1) OF STD_LOGIC_VECTOR(15 DOWNTO 0);
    SIGNAL img_mem : mem_t := (OTHERS => (OTHERS => '0'));
    SIGNAL mem_read_lat : STD_LOGIC_VECTOR(15 DOWNTO 0) := (OTHERS => '0');

    TYPE omem_t IS ARRAY (0 TO 8100-1) OF STD_LOGIC_VECTOR(15 DOWNTO 0);
    SIGNAL out_mem : omem_t := (OTHERS => (OTHERS => '0'));

    SIGNAL file_loaded : BOOLEAN := false;
    SIGNAL sim_done    : BOOLEAN := false;

BEGIN

    dut : HESSIAN_conv_c
    PORT MAP(
        clk           => clk,
        trg           => trg,
        rdy           => rdy,
        conv_in_en    => conv_in_en,
        conv_in_addr  => conv_in_addr,
        conv_in_d     => conv_in_d,
        conv_out_en   => conv_out_en,
        conv_out_addr => conv_out_addr,
        conv_out_d    => conv_out_d
    );

    clk_process : PROCESS
    BEGIN
        WHILE NOT sim_done LOOP
            clk <= '0';
            WAIT FOR clk_period/2;
            clk <= '1';
            WAIT FOR clk_period/2;
        END LOOP;
        WAIT;
    END PROCESS;

    mem_read_process : PROCESS(clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF conv_in_en = '1' THEN
                mem_read_lat <= img_mem(to_integer(unsigned(conv_in_addr)));
            END IF;
        END IF;
    END PROCESS;

    conv_in_d <= mem_read_lat;

    out_write_process : PROCESS(clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF conv_out_en = '1' THEN
                out_mem(to_integer(unsigned(conv_out_addr))) <= conv_out_d;
            END IF;
        END IF;
    END PROCESS;

    load_file : PROCESS
        FILE f        : text;
        VARIABLE l    : line;
        VARIABLE val  : INTEGER;
        VARIABLE idx  : INTEGER := 0;
        VARIABLE status : file_open_status;
    BEGIN
        file_open(status, f, INPUT_FILE, read_mode);

        IF status /= open_ok THEN
            REPORT "ERROR opening conv_r input file" SEVERITY FAILURE;
        END IF;

        WHILE NOT endfile(f) LOOP
            readline(f, l);
            read(l, val);
            img_mem(idx) <= STD_LOGIC_VECTOR(to_signed(val, 16));
            idx := idx + 1;
        END LOOP;

        file_close(f);
        REPORT "Loaded " & INTEGER'image(idx) & " pixels of stage03 conv_r output.";

        file_loaded <= true;
        WAIT;
    END PROCESS;

    stim : PROCESS
    BEGIN
        WAIT UNTIL file_loaded = true;
        WAIT FOR 100 ns;

        REPORT "Starting HESSIAN column convolution...";

        WAIT UNTIL rising_edge(clk);
        trg <= '1';
        WAIT UNTIL rising_edge(clk);
        trg <= '0';

        WAIT UNTIL rdy = '1';

        REPORT "Hessian column convolution finished.";

        sim_done <= true;
        WAIT;
    END PROCESS;

    save_output : PROCESS
        FILE ofile : text;
        VARIABLE l  : line;
        VARIABLE tmp : INTEGER;
    BEGIN
        WAIT UNTIL rdy = '1';
        WAIT FOR 20 ns;

        file_open(ofile, OUTPUT_FILE, write_mode);

        FOR i IN 0 TO 8099 LOOP
            tmp := to_integer(signed(out_mem(i)));
            write(l, tmp);
            writeline(ofile, l);
        END LOOP;

        file_close(ofile);
        REPORT "Column convolution output saved.";

        WAIT;
    END PROCESS;

END arch;
