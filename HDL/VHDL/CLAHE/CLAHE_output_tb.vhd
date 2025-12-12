LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;
USE STD.textio.ALL;
USE IEEE.std_logic_textio.ALL;

ENTITY CLAHE_output_tb IS
END CLAHE_output_tb;

ARCHITECTURE behavior OF CLAHE_output_tb IS

    -- Component Declaration
    COMPONENT CLAHE_output
        PORT (
            clk : IN STD_LOGIC;
            trg : IN STD_LOGIC;
            rdy : OUT STD_LOGIC;
            img_in_en : OUT STD_LOGIC;
            img_in_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            img_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            mapping_in_en : OUT STD_LOGIC;
            mapping_in_addr : OUT STD_LOGIC_VECTOR(11 DOWNTO 0);
            mapping_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            clahe_out_en : OUT STD_LOGIC;
            clahe_out_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
            clahe_out_d : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
        );
    END COMPONENT;

    -- Clock period
    CONSTANT clk_period : TIME := 10 ns;

    -- Testbench signals
    SIGNAL clk : STD_LOGIC := '0';
    SIGNAL trg : STD_LOGIC := '0';
    SIGNAL rdy : STD_LOGIC;
    SIGNAL img_in_en : STD_LOGIC;
    SIGNAL img_in_addr : STD_LOGIC_VECTOR(13 DOWNTO 0);
    SIGNAL img_in_d : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL mapping_in_en : STD_LOGIC;
    SIGNAL mapping_in_addr : STD_LOGIC_VECTOR(11 DOWNTO 0);
    SIGNAL mapping_in_d : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL clahe_out_en : STD_LOGIC;
    SIGNAL clahe_out_addr : STD_LOGIC_VECTOR(13 DOWNTO 0);
    SIGNAL clahe_out_d : STD_LOGIC_VECTOR(7 DOWNTO 0);

    -- Memory arrays
    TYPE img_mem_type IS ARRAY (0 TO 16383) OF STD_LOGIC_VECTOR(7 DOWNTO 0); -- 128*128 input
    TYPE map_mem_type IS ARRAY (0 TO 4095) OF STD_LOGIC_VECTOR(7 DOWNTO 0); -- 16*256 mappings
    TYPE out_mem_type IS ARRAY (0 TO 9215) OF STD_LOGIC_VECTOR(7 DOWNTO 0); -- 96*96 output

    SIGNAL img_memory : img_mem_type := (OTHERS => (OTHERS => '0'));
    SIGNAL map_memory : map_mem_type := (OTHERS => (OTHERS => '0'));
    SIGNAL out_memory : out_mem_type := (OTHERS => (OTHERS => '0'));

    SIGNAL sim_done : BOOLEAN := false;

BEGIN

    -- Instantiate the Unit Under Test (UUT)
    uut : CLAHE_output
    PORT MAP(
        clk => clk,
        trg => trg,
        rdy => rdy,
        img_in_en => img_in_en,
        img_in_addr => img_in_addr,
        img_in_d => img_in_d,
        mapping_in_en => mapping_in_en,
        mapping_in_addr => mapping_in_addr,
        mapping_in_d => mapping_in_d,
        clahe_out_en => clahe_out_en,
        clahe_out_addr => clahe_out_addr,
        clahe_out_d => clahe_out_d
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

    -- Input Image RAM process (read only)
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

    -- Mapping RAM process (read only)
    map_ram_process : PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF mapping_in_en = '1' THEN
                mapping_in_d <= map_memory(to_integer(unsigned(mapping_in_addr)));
            ELSE
                mapping_in_d <= (OTHERS => '0');
            END IF;
        END IF;
    END PROCESS;

    -- Output RAM process (write only)
    out_ram_process : PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF clahe_out_en = '1' THEN
                out_memory(to_integer(unsigned(clahe_out_addr))) <= clahe_out_d;
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
        trg <= '0';

        REPORT "========================================" SEVERITY NOTE;
        REPORT "CLAHE Output Testbench Started" SEVERITY NOTE;
        REPORT "========================================" SEVERITY NOTE;

        -- Load original input image
        REPORT "Loading original image from input_to_clahe.txt..." SEVERITY NOTE;
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
        REPORT "Loaded " & INTEGER'image(addr) & " pixels from original image" SEVERITY NOTE;

        -- Load mapping tables from CLAHE_mappings output
        REPORT "Loading mapping tables from output_mappings_all.txt..." SEVERITY NOTE;
        file_open(input_file, "output_mappings_all.txt", read_mode);
        addr := 0;
        WHILE NOT endfile(input_file) AND addr < 4096 LOOP
            readline(input_file, file_line);
            read(file_line, pixel_val, success);
            IF success THEN
                map_memory(addr) <= STD_LOGIC_VECTOR(to_unsigned(pixel_val, 8));
                addr := addr + 1;
            END IF;
        END LOOP;
        file_close(input_file);
        REPORT "Loaded " & INTEGER'image(addr) & " mapping values (16 patches)" SEVERITY NOTE;

        IF addr < 4096 THEN
            REPORT "ERROR: Not enough mapping data loaded!" SEVERITY ERROR;
            REPORT "Expected 4096 values, got " & INTEGER'image(addr) SEVERITY ERROR;
            sim_done <= true;
            WAIT;
        END IF;

        -- Wait for system to stabilize
        WAIT FOR 100 ns;

        -- Start CLAHE output processing
        REPORT "========================================" SEVERITY NOTE;
        REPORT "Starting CLAHE output interpolation (96x96)..." SEVERITY NOTE;
        REPORT "========================================" SEVERITY NOTE;

        start_time := NOW;
        trg <= '1';
        WAIT FOR clk_period * 2;
        trg <= '0';

        -- Wait for completion
        WAIT UNTIL rdy = '1';
        end_time := NOW;

        WAIT FOR clk_period * 10;

        REPORT "========================================" SEVERITY NOTE;
        REPORT "CLAHE output processing complete!" SEVERITY NOTE;
        REPORT "Processing time: " & TIME'image(end_time - start_time) SEVERITY NOTE;
        REPORT "========================================" SEVERITY NOTE;

        -- Save enhanced output image (96x96 = 9216 pixels)
        REPORT "Saving enhanced image to new_image.txt..." SEVERITY NOTE;

        file_open(output_file, "new_image.txt", write_mode);

        FOR i IN 0 TO 9215 LOOP -- 96*96 - 1
            write(file_line, to_integer(unsigned(out_memory(i))));
            writeline(output_file, file_line);
        END LOOP;

        file_close(output_file);
        REPORT "Saved 9216 pixels (96x96) to new_image.txt" SEVERITY NOTE;

        -- Print sample values
        REPORT "========================================" SEVERITY NOTE;
        REPORT "Sample output values (first 16 pixels):" SEVERITY NOTE;
        FOR i IN 0 TO 15 LOOP
            REPORT "  pixel[" & INTEGER'image(i) & "] = " &
                INTEGER'image(to_integer(unsigned(out_memory(i)))) SEVERITY NOTE;
        END LOOP;
        REPORT "========================================" SEVERITY NOTE;

        -- Verify non-zero output
        addr := 0;
        FOR i IN 0 TO 9215 LOOP
            IF out_memory(i) /= x"00" THEN
                addr := addr + 1;
            END IF;
        END LOOP;

        REPORT "Non-zero output pixels: " & INTEGER'image(addr) & " / 9216" SEVERITY NOTE;

        IF addr > 0 THEN
            REPORT "========================================" SEVERITY NOTE;
            REPORT "TEST PASSED: Enhanced image generated!" SEVERITY NOTE;
            REPORT "========================================" SEVERITY NOTE;
        ELSE
            REPORT "========================================" SEVERITY ERROR;
            REPORT "TEST FAILED: All output pixels are zero!" SEVERITY ERROR;
            REPORT "========================================" SEVERITY ERROR;
        END IF;

        -- End simulation
        sim_done <= true;
        REPORT "Simulation finished. Check new_image.txt for enhanced image." SEVERITY NOTE;
        WAIT;
    END PROCESS;

END behavior;