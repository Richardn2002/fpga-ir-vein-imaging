LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;
USE STD.textio.ALL;
USE IEEE.std_logic_textio.ALL;

ENTITY CLAHE_controller_tb IS
END CLAHE_controller_tb;

ARCHITECTURE behavior OF CLAHE_controller_tb IS

    -- Component Declaration
    COMPONENT CLAHE_controller
        PORT (
            clk : IN STD_LOGIC;
            start : IN STD_LOGIC;
            done : OUT STD_LOGIC;
            img_in_en : OUT STD_LOGIC;
            img_in_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            img_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            mapping_ren : OUT STD_LOGIC;
            mapping_wen : OUT STD_LOGIC;
            mapping_addr : OUT STD_LOGIC_VECTOR(11 DOWNTO 0);
            mapping_din : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            mapping_dout : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
        );
    END COMPONENT;

    -- Clock period
    CONSTANT clk_period : TIME := 10 ns;

    -- Testbench signals
    SIGNAL clk : STD_LOGIC := '0';
    SIGNAL start : STD_LOGIC := '0';
    SIGNAL done : STD_LOGIC;
    SIGNAL img_in_en : STD_LOGIC;
    SIGNAL img_in_addr : STD_LOGIC_VECTOR(13 DOWNTO 0);
    SIGNAL img_in_d : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL mapping_ren : STD_LOGIC;
    SIGNAL mapping_wen : STD_LOGIC;
    SIGNAL mapping_addr : STD_LOGIC_VECTOR(11 DOWNTO 0);
    SIGNAL mapping_din : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL mapping_dout : STD_LOGIC_VECTOR(7 DOWNTO 0);

    -- Memory arrays
    TYPE img_mem_type IS ARRAY (0 TO 16383) OF STD_LOGIC_VECTOR(7 DOWNTO 0); -- 128*128
    TYPE map_mem_type IS ARRAY (0 TO 4095) OF STD_LOGIC_VECTOR(7 DOWNTO 0); -- 16*256

    SIGNAL img_memory : img_mem_type := (OTHERS => (OTHERS => '0'));
    SIGNAL map_memory : map_mem_type := (OTHERS => (OTHERS => '0'));

    SIGNAL sim_done : BOOLEAN := false;

BEGIN

    -- Instantiate the Unit Under Test (UUT)
    uut : CLAHE_controller
    PORT MAP(
        clk => clk,
        start => start,
        done => done,
        img_in_en => img_in_en,
        img_in_addr => img_in_addr,
        img_in_d => img_in_d,
        mapping_ren => mapping_ren,
        mapping_wen => mapping_wen,
        mapping_addr => mapping_addr,
        mapping_din => mapping_din,
        mapping_dout => mapping_dout
    );

    -- Clock process
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

    -- Image RAM process (read only, 1 cycle latency)
    img_ram_process : PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF img_in_en = '1' THEN
                img_in_d <= img_memory(to_integer(unsigned(img_in_addr)));
            ELSE
                img_in_d <= (OTHERS => '0');
            END IF;
        END IF;
    END PROCESS;

    -- Mapping RAM process (read/write, 1 cycle latency)
    map_ram_process : PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            -- Write has priority
            IF mapping_wen = '1' THEN
                map_memory(to_integer(unsigned(mapping_addr))) <= mapping_dout;
            END IF;

            -- Read
            IF mapping_ren = '1' THEN
                mapping_din <= map_memory(to_integer(unsigned(mapping_addr)));
            ELSE
                mapping_din <= (OTHERS => '0');
            END IF;
        END IF;
    END PROCESS;

    -- Stimulus process
    stim_proc : PROCESS
        FILE input_file : text;
        FILE output_file : text;
        VARIABLE file_line : line;
        VARIABLE pixel_val : INTEGER;
        VARIABLE addr : INTEGER;
        VARIABLE success : BOOLEAN;
        VARIABLE start_time : TIME;
        VARIABLE end_time : TIME;
    BEGIN
        -- Initialize Inputs
        start <= '0';

        REPORT "========================================" SEVERITY NOTE;
        REPORT "CLAHE Controller Testbench Started" SEVERITY NOTE;
        REPORT "========================================" SEVERITY NOTE;

        -- Initialize image memory from file
        REPORT "Loading image from input_to_clahe.txt..." SEVERITY NOTE;

        file_open(input_file, "input_to_CLAHE.txt", read_mode);
        addr := 0;
        WHILE NOT endfile(input_file) AND addr < 16384 LOOP
            readline(input_file, file_line);
            read(file_line, pixel_val, success);
            IF success THEN
                img_memory(addr) <= STD_LOGIC_VECTOR(to_unsigned(pixel_val, 8));
                addr := addr + 1;
            END IF;
        END LOOP;
        file_close(input_file);

        REPORT "Loaded " & INTEGER'image(addr) & " pixels from file" SEVERITY NOTE;

        -- If file doesn't exist or is empty, generate test pattern
        IF addr = 0 THEN
            REPORT "File not found. Generating test pattern..." SEVERITY NOTE;
            FOR i IN 0 TO 16383 LOOP
                -- Low contrast pattern (values 100-150)
                img_memory(i) <= STD_LOGIC_VECTOR(to_unsigned(100 + (i MOD 51), 8));
            END LOOP;
            REPORT "Generated 16384 test pixels" SEVERITY NOTE;
        END IF;

        -- Wait for system to stabilize
        WAIT FOR 100 ns;

        -- Start CLAHE processing
        REPORT "========================================" SEVERITY NOTE;
        REPORT "Starting CLAHE processing for all 16 patches..." SEVERITY NOTE;
        REPORT "========================================" SEVERITY NOTE;

        start_time := NOW;
        start <= '1';
        WAIT FOR clk_period * 2;
        start <= '0';

        -- Wait for completion
        WAIT UNTIL done = '1';
        end_time := NOW;

        WAIT FOR clk_period * 10;

        REPORT "========================================" SEVERITY NOTE;
        REPORT "CLAHE processing complete!" SEVERITY NOTE;
        REPORT "Processing time: " & TIME'image(end_time - start_time) SEVERITY NOTE;
        REPORT "========================================" SEVERITY NOTE;

        -- Save all 16 mapping tables to one file
        REPORT "Saving all mapping tables to output_mappings_all.txt..." SEVERITY NOTE;

        file_open(output_file, "output_mappings_all.txt", write_mode);

        FOR patch IN 0 TO 15 LOOP
            FOR i IN 0 TO 255 LOOP
                addr := patch * 256 + i;
                write(file_line, to_integer(unsigned(map_memory(addr))));
                writeline(output_file, file_line);
            END LOOP;
        END LOOP;

        file_close(output_file);
        REPORT "Saved all 16 patches (4096 values) to output_mappings_all.txt" SEVERITY NOTE;

        -- Print statistics for patch 0
        REPORT "========================================" SEVERITY NOTE;
        REPORT "Sample: Patch 0 mapping values (first 16):" SEVERITY NOTE;
        FOR i IN 0 TO 15 LOOP
            REPORT "  mapping[" & INTEGER'image(i) & "] = " &
                INTEGER'image(to_integer(unsigned(map_memory(i)))) SEVERITY NOTE;
        END LOOP;
        REPORT "========================================" SEVERITY NOTE;

        -- Verify non-zero mappings
        addr := 0;
        FOR i IN 0 TO 4095 LOOP
            IF map_memory(i) /= x"00" THEN
                addr := addr + 1;
            END IF;
        END LOOP;

        REPORT "Non-zero mapping entries: " & INTEGER'image(addr) & " / 4096" SEVERITY NOTE;

        IF addr > 0 THEN
            REPORT "========================================" SEVERITY NOTE;
            REPORT "TEST PASSED: Mappings generated successfully!" SEVERITY NOTE;
            REPORT "========================================" SEVERITY NOTE;
        ELSE
            REPORT "========================================" SEVERITY ERROR;
            REPORT "TEST FAILED: All mappings are zero!" SEVERITY ERROR;
            REPORT "========================================" SEVERITY ERROR;
        END IF;

        -- End simulation
        sim_done <= true;
        REPORT "Simulation finished. Check output_mapping_patch*.txt files." SEVERITY NOTE;
        WAIT;
    END PROCESS;

END behavior;