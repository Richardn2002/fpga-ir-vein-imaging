LIBRARY IEEE;
USE ieee.std_logic_1164.ALL;

ENTITY cdc_sync IS
    -- metastability sync a signal from a slow clk domain to a fast clk domain
    -- metastability sync a trigger from a fast clk domain to a slow clk domain
    GENERIC (
        N_SYNC : NATURAL := 2
    );
    PORT (
        clk_slow : IN STD_LOGIC;
        sig_from_slow : IN STD_LOGIC;
        trg_to_slow : OUT STD_LOGIC;

        clk_fast : IN STD_LOGIC;
        trg_from_fast : IN STD_LOGIC;
        sig_to_fast : OUT STD_LOGIC
    );
END ENTITY;

ARCHITECTURE arch OF cdc_sync IS
    ATTRIBUTE ASYNC_REG : STRING;

    SIGNAL sig_from_slow_sync : STD_LOGIC_VECTOR(N_SYNC - 1 DOWNTO 0) := (OTHERS => '0');
    ATTRIBUTE ASYNC_REG OF sig_from_slow_sync : SIGNAL IS "TRUE";

    SIGNAL trg_from_fast_tgl : STD_LOGIC := '0';
    SIGNAL trg_from_fast_tgl_sync : STD_LOGIC_VECTOR(N_SYNC - 1 DOWNTO 0) := (OTHERS => '0');
    ATTRIBUTE ASYNC_REG OF trg_from_fast_tgl_sync : SIGNAL IS "TRUE";
    SIGNAL trg_from_fast_tgl_sync_last : STD_LOGIC := '0';
BEGIN
    PROCESS (clk_fast) BEGIN
        IF rising_edge(clk_fast) THEN
            sig_from_slow_sync <= sig_from_slow_sync(N_SYNC - 2 DOWNTO 0) & sig_from_slow;
            sig_to_fast <= sig_from_slow_sync(N_SYNC - 1);

            IF trg_from_fast = '1' THEN
                trg_from_fast_tgl <= NOT trg_from_fast_tgl;
            END IF;
        END IF;
    END PROCESS;

    PROCESS (clk_slow)
    BEGIN
        IF rising_edge(clk_slow) THEN
            trg_from_fast_tgl_sync <= trg_from_fast_tgl_sync(N_SYNC - 2 DOWNTO 0) & trg_from_fast_tgl;
            trg_from_fast_tgl_sync_last <= trg_from_fast_tgl_sync(N_SYNC - 1);
            trg_to_slow <= trg_from_fast_tgl_sync_last XOR trg_from_fast_tgl_sync(N_SYNC - 1);
        END IF;
    END PROCESS;
END ARCHITECTURE;