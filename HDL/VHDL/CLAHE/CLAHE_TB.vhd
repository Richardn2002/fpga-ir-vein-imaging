LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;
USE STD.textio.ALL;
USE IEEE.std_logic_textio.ALL;

ENTITY tb_CLAHE_mappings IS
    GENERIC (
        -- CHANGE THESE PATHS IF NEEDED
        INPUT_FILE_NAME : STRING := "/raid/fall2025/wlu33/FPGA/Project/gray_o_clean.txt";
        OUTPUT_MAP_FILE : STRING := "clahe_mapping_table.txt";
        OUTPUT_IMG_FILE : STRING := "clahe_output_image.txt"
    );
END tb_CLAHE_mappings;

ARCHITECTURE arch OF tb_CLAHE_mappings IS

    -- Component Declaration
    COMPONENT CLAHE_mappings
        PORT (
            clk : IN STD_LOGIC;
            trg : IN STD_LOGIC;
            rdy : OUT STD_LOGIC;
            img_in_en : OUT STD_LOGIC;
            img_in_addr : OUT STD_LOGIC_VECTOR(9 DOWNTO 0);
            img_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            hist_mapping_inout_ren : OUT STD_LOGIC;
            hist_mapping_inout_wen : OUT STD_LOGIC;
            hist_mapping_inout_addr : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
            hist_mapping_inout_din : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            hist_mapping_inout_dout : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
        );
    END COMPONENT;

    -- Clock period
    CONSTANT clk_period : TIME := 10 ns;

    -- Signals
    SIGNAL clk : STD_LOGIC := '0';
    SIGNAL trg : STD_LOGIC := '0';
    SIGNAL rdy : STD_LOGIC;

    -- DUT Interface Signals
    SIGNAL img_in_en : STD_LOGIC;
    SIGNAL img_in_addr : STD_LOGIC_VECTOR(9 DOWNTO 0);
    SIGNAL img_in_d : STD_LOGIC_VECTOR(7 DOWNTO 0) := (OTHERS => '0');

    SIGNAL hist_mapping_inout_ren : STD_LOGIC;
    SIGNAL hist_mapping_inout_wen : STD_LOGIC;
    SIGNAL hist_mapping_inout_addr : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL hist_mapping_inout_din : STD_LOGIC_VECTOR(7 DOWNTO 0) := (OTHERS => '0');
    SIGNAL hist_mapping_inout_dout : STD_LOGIC_VECTOR(7 DOWNTO 0);

    -- Helper signal for the synchronous read (MOVED HERE)
    SIGNAL img_memory_read : STD_LOGIC_VECTOR(7 DOWNTO 0) := (OTHERS => '0');

    -- Memory arrays
    TYPE img_mem_type IS ARRAY (0 TO 1023) OF STD_LOGIC_VECTOR(7 DOWNTO 0);
    TYPE hist_mem_type IS ARRAY (0 TO 255) OF STD_LOGIC_VECTOR(7 DOWNTO 0);

    SIGNAL img_memory : img_mem_type := (OTHERS => (OTHERS => '0'));
    SIGNAL hist_memory : hist_mem_type := (OTHERS => (OTHERS => '0'));

    -- Test control flags
    SIGNAL test_done : BOOLEAN := false;
    SIGNAL image_loaded : BOOLEAN := false;

BEGIN

    -- 1. Instantiate Unit Under Test
    uut : CLAHE_mappings
    PORT MAP(
        clk => clk,
        trg => trg,
        rdy => rdy,
        img_in_en => img_in_en,
        img_in_addr => img_in_addr,
        img_in_d => img_in_d,
        hist_mapping_inout_ren => hist_mapping_inout_ren,
        hist_mapping_inout_wen => hist_mapping_inout_wen,
        hist_mapping_inout_addr => hist_mapping_inout_addr,
        hist_mapping_inout_din => hist_mapping_inout_din,
        hist_mapping_inout_dout => hist_mapping_inout_dout
    );

    -- 2. Clock Generation
    clk_process : PROCESS
    BEGIN
        WHILE NOT test_done LOOP
            clk <= '0';
            WAIT FOR clk_period/2;
            clk <= '1';
            WAIT FOR clk_period/2;
        END LOOP;
        WAIT;
    END PROCESS;

    -- 3. Image Memory (Read Only)
    -- Modeled as Synchronous RAM (1 cycle latency)
    img_mem_read : PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF img_in_en = '1' THEN
                img_memory_read <= img_memory(to_integer(unsigned(img_in_addr)));
            END IF;
        END IF;
    END PROCESS;

    -- Assign the read value to the DUT input
    img_in_d <= img_memory_read;

    -- 4. Histogram/Mapping Memory (Read/Write)
    -- This RAM is shared: The DUT calculates the map and writes it here.
    hist_mem_process : PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            -- Write port (High Priority)
            IF hist_mapping_inout_wen = '1' THEN
                hist_memory(to_integer(unsigned(hist_mapping_inout_addr))) <= hist_mapping_inout_dout;
            END IF;

            -- Read port
            IF hist_mapping_inout_ren = '1' THEN
                hist_mapping_inout_din <= hist_memory(to_integer(unsigned(hist_mapping_inout_addr)));
            END IF;
        END IF;
    END PROCESS;

    -- 5. Load Image from File
    load_image : PROCESS
        FILE img_file : text;
        VARIABLE img_line : line;
        VARIABLE pixel_val : INTEGER;
        VARIABLE addr : INTEGER := 0;
        VARIABLE file_status : file_open_status;
    BEGIN
        file_open(file_status, img_file, INPUT_FILE_NAME, read_mode);

        IF file_status /= open_ok THEN
            REPORT "ERROR: Cannot open file " & INPUT_FILE_NAME SEVERITY FAILURE;
        END IF;

        WHILE NOT endfile(img_file) AND addr < 1024 LOOP
            readline(img_file, img_line);
            IF img_line'length > 0 THEN
                read(img_line, pixel_val);
                img_memory(addr) <= STD_LOGIC_VECTOR(to_unsigned(pixel_val, 8));
                addr := addr + 1;
            END IF;
        END LOOP;

        file_close(img_file);
        REPORT "Loaded " & INTEGER'image(addr) & " pixels.";
        image_loaded <= true;
        WAIT;
    END PROCESS;

    -- 6. Main Stimulus Process
    stim_proc : PROCESS
    BEGIN
        -- Wait for load
        WAIT UNTIL image_loaded = true;
        WAIT FOR clk_period * 10;

        REPORT "Starting CLAHE processing...";

        -- Pulse Trigger
        WAIT UNTIL rising_edge(clk);
        trg <= '1';
        WAIT FOR clk_period;
        trg <= '0';

        -- WAIT FOR CALCULATION TO FINISH
        WAIT UNTIL rdy = '1';

        REPORT "CLAHE Calculation Finished. RAM now contains the Mapping.";
        WAIT FOR clk_period * 10;

        -- Signal end of simulation (allows file saving to finish)
        test_done <= true;
        WAIT;
    END PROCESS;

    -- 7. Apply Map & Save Results
    save_results : PROCESS
        FILE map_file : text;
        FILE out_img_file : text;
        VARIABLE map_line : line;
        VARIABLE out_line : line;
        VARIABLE pixel_val : INTEGER;
        VARIABLE mapped_val : INTEGER;
    BEGIN
        -- Wait until hardware says it is done
        WAIT UNTIL rdy = '1';
        WAIT FOR clk_period * 5;

        -- A. Save the Histogram/Map for debugging
        file_open(map_file, OUTPUT_MAP_FILE, write_mode);
        FOR i IN 0 TO 255 LOOP
            write(map_line, to_integer(unsigned(hist_memory(i))));
            writeline(map_file, map_line);
        END LOOP;
        file_close(map_file);

        -- B. Apply the Map to the Image
        file_open(out_img_file, OUTPUT_IMG_FILE, write_mode);

        FOR addr IN 0 TO 1023 LOOP
            -- 1. Get original pixel
            pixel_val := to_integer(unsigned(img_memory(addr)));

            -- 2. Look up the new value in the LUT created by the Hardware
            mapped_val := to_integer(unsigned(hist_memory(pixel_val)));

            -- 3. Write to file
            write(out_line, mapped_val);
            writeline(out_img_file, out_line);
        END LOOP;

        file_close(out_img_file);
        REPORT "Output files saved successfully.";
        WAIT;
    END PROCESS;

END arch;