LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

-- Top-level controller that processes all 16 patches sequentially
ENTITY CLAHE_controller IS
    PORT (
        clk : IN STD_LOGIC;
        start : IN STD_LOGIC;
        done : OUT STD_LOGIC;

        -- Image RAM Interface (shared)
        img_in_en : OUT STD_LOGIC;
        img_in_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
        img_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        -- Mapping RAM Interface (shared)
        mapping_ren : OUT STD_LOGIC;
        mapping_wen : OUT STD_LOGIC;
        mapping_addr : OUT STD_LOGIC_VECTOR(11 DOWNTO 0);
        mapping_din : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        mapping_dout : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END CLAHE_controller;

ARCHITECTURE arch OF CLAHE_controller IS

    -- State machine
    TYPE ctrl_state_type IS (CTRL_IDLE, CTRL_PROCESSING, CTRL_WAIT, CTRL_DONE);
    SIGNAL ctrl_state : ctrl_state_type := CTRL_IDLE;
    ATTRIBUTE fsm_safe_state : STRING;
    ATTRIBUTE fsm_safe_state OF ctrl_state : SIGNAL IS "power_on_state";

    SIGNAL current_patch : INTEGER RANGE 0 TO 15 := 0;

    -- Arrays for all 16 patch instances
    TYPE trg_array IS ARRAY (0 TO 15) OF STD_LOGIC;
    TYPE rdy_array IS ARRAY (0 TO 15) OF STD_LOGIC;
    TYPE en_array IS ARRAY (0 TO 15) OF STD_LOGIC;
    TYPE addr14_array IS ARRAY (0 TO 15) OF STD_LOGIC_VECTOR(13 DOWNTO 0);
    TYPE addr12_array IS ARRAY (0 TO 15) OF STD_LOGIC_VECTOR(11 DOWNTO 0);
    TYPE data_array IS ARRAY (0 TO 15) OF STD_LOGIC_VECTOR(7 DOWNTO 0);

    SIGNAL patch_trg : trg_array := (OTHERS => '0');
    SIGNAL patch_rdy : rdy_array;
    SIGNAL patch_img_en : en_array;
    SIGNAL patch_img_addr : addr14_array;
    SIGNAL patch_map_ren : en_array;
    SIGNAL patch_map_wen : en_array;
    SIGNAL patch_map_addr : addr12_array;
    SIGNAL patch_map_dout : data_array;

BEGIN

    -- Generate 16 instances of CLAHE_mappings
    gen_patches : FOR i IN 0 TO 15 GENERATE
        patch_inst : ENTITY work.CLAHE_mappings
            GENERIC MAP(
                PATCH_IDX => i
            )
            PORT MAP(
                clk => clk,
                trg => patch_trg(i),
                rdy => patch_rdy(i),
                img_in_en => patch_img_en(i),
                img_in_addr => patch_img_addr(i),
                img_in_d => img_in_d,
                hist_mapping_inout_ren => patch_map_ren(i),
                hist_mapping_inout_wen => patch_map_wen(i),
                hist_mapping_inout_addr => patch_map_addr(i),
                hist_mapping_inout_din => mapping_din,
                hist_mapping_inout_dout => patch_map_dout(i)
            );
    END GENERATE;

    -- Multiplex outputs from current active patch
    img_in_en <= patch_img_en(current_patch);
    img_in_addr <= patch_img_addr(current_patch);
    mapping_ren <= patch_map_ren(current_patch);
    mapping_wen <= patch_map_wen(current_patch);
    mapping_addr <= patch_map_addr(current_patch);
    mapping_dout <= patch_map_dout(current_patch);

    -- Control FSM
    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            CASE ctrl_state IS
                WHEN CTRL_IDLE =>
                    done <= '0';
                    current_patch <= 0;

                    IF start = '1' THEN
                        patch_trg(0) <= '1'; -- Trigger first patch
                        ctrl_state <= CTRL_PROCESSING;
                    END IF;

                WHEN CTRL_PROCESSING =>
                    -- Clear trigger after one cycle
                    patch_trg(current_patch) <= '0';
                    ctrl_state <= CTRL_WAIT;

                WHEN CTRL_WAIT =>
                    IF patch_rdy(current_patch) = '1' THEN
                        IF current_patch = 15 THEN
                            -- All patches done
                            ctrl_state <= CTRL_DONE;
                        ELSE
                            -- Move to next patch
                            current_patch <= current_patch + 1;
                            patch_trg(current_patch + 1) <= '1';
                            ctrl_state <= CTRL_PROCESSING;
                        END IF;
                    END IF;

                WHEN CTRL_DONE =>
                    done <= '1';
                    IF start = '0' THEN
                        ctrl_state <= CTRL_IDLE;
                    END IF;

                WHEN OTHERS =>
                    ctrl_state <= CTRL_IDLE;
            END CASE;
        END IF;
    END PROCESS;

END arch;