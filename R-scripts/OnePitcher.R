# ---- Configuration ----
input_file_path <- "WSH"  # Team code (e.g., "PIT", "WSH") or full path
output_directory <- path.expand("~/Downloads/")

# ---- Optional Filtering Parameters ----
# Set these to NULL to disable filtering
selected_pitcher_filter <- "Abner, Philip"  # Example format: "Bieber, Shane"
start_date_filter <- "2026-01-01"        # Example format: "2025-04-01"
end_date_filter <- "2026-11-28"         # Example format: "2025-04-20"

# ---- Required Libraries ----
library(tidyverse)
library(patchwork)
library(gridExtra)
library(gtable)
library(grid)
library(cowplot)

# Source shared utilities (pitch names/colors/order, data loading, stat
# summaries, table formatting, pitch plot, and the shared report driver).
# Resolve the utils path from this script's own location so the scripts can be
# run from any working directory; fall back to the repo's canonical location.
.script_dir <- local({
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
  } else {
    path.expand("~/Huronalytics/R-scripts")
  }
})
source(file.path(.script_dir, "pitcher_report_utils.R"))

# Allow command-line overrides: Rscript OnePitcher.R "/path/to/file.csv" "2026-04-08" "2026-04-08" "Penrod, Zach"
cli_args <- commandArgs(trailingOnly = TRUE)
if (length(cli_args) >= 1) input_file_path <- cli_args[1]
if (length(cli_args) >= 2) start_date_filter <- cli_args[2]
if (length(cli_args) >= 3) end_date_filter <- cli_args[3]
if (length(cli_args) >= 4) selected_pitcher_filter <- cli_args[4]

# Resolve team code to full path (e.g., "PIT" -> "~/Downloads/NLC2026 - PIT.csv")
input_file_path <- resolve_team_path(input_file_path)

# ---- Visualization Functions ----

# Pitch movement plot: titled, no arm-angle rays
create_pitch_plot <- function(pitch_data_filtered, pitcher_name, game_date = NULL) {
  create_pitch_plot_shared(pitch_data_filtered, pitcher_name, game_date,
                           arm_angle_lines = FALSE, show_title = TRUE)
}

# Stats tables: pitch-metric table (Table 1) over outcome table (Table 2)
create_pitcher_tables <- function(pitch_data, selected_pitcher, game_date = NULL) {
  # Filter by date if provided
  if (!is.null(game_date)) {
    pitch_data <- pitch_data %>%
      filter(`Game Date` == game_date)
  }

  # Check if Arm Angle column exists and has data for this pitcher
  has_arm_angle <- col_has_data(pitch_data, selected_pitcher, "ArmAngle")

  # One shared summary feeds both tables
  stats_full <- summarize_pitch_type_stats(pitch_data, selected_pitcher,
                                           has_arm_angle, gb_zero = "---")

  # Handle case where no stats are available
  if (nrow(stats_full) == 0) {
    return(grid.text("No pitch data available", gp = gpar(fontsize = 16, fontface = "bold")))
  }

  # FIRST TABLE - pitch metrics. keep_populated_cols drops any column whose
  # source is empty for the whole outing (Arm Angle included, so has_arm_angle
  # no longer needs its own branch here).
  core_cols <- c(
    `Pitch Type` = "Pitch Type", num_thrown = "Count", percent_thrown = "% Thrown",
    avg_velo = "Velocity", max_velo = "Max Velo", avg_spin = "Spin Rate",
    avg_rtilt = "RTilt", avg_tilt = "OTilt", avg_ivb = "IVB", avg_hb = "HB",
    avg_height = "RelZ", avg_side = "RelX", avg_extension = "Ext",
    avg_arm_angle = "Arm Angle", avg_vaa = "VAA", avg_haa = "HAA"
  )
  core_cols <- core_cols[keep_populated_cols(names(core_cols), pitch_data, selected_pitcher)]
  stats_df_table1 <- map_and_sort_pitch_types(stats_full %>% select(all_of(names(core_cols))))
  names(stats_df_table1) <- unname(core_cols)

  # SECOND TABLE - outcome metrics
  outcome_cols <- c(
    `Pitch Type` = "Pitch Type", num_thrown = "Count", percent_thrown = "% Thrown",
    iz_percent = "Zone%", swing_percent = "Swing%", csw_percent = "CSW%",
    swstr_percent = "Whiff%", chase_percent = "Chase%", gb_percent = "GB%"
  )
  outcome_cols <- outcome_cols[keep_populated_cols(names(outcome_cols), pitch_data, selected_pitcher)]
  stats_df_table2 <- map_and_sort_pitch_types(stats_full %>% select(all_of(names(outcome_cols))))
  names(stats_df_table2) <- unname(outcome_cols)

  # Create base table theme with larger font; tight horizontal padding so each
  # column is only as wide as its widest content
  tt <- make_table_theme(core_fontsize = 16, head_fontsize = 16, pad_h_mm = 8)

  # Create both tables (column widths auto-size to fit content)
  tbl1 <- tableGrob(stats_df_table1, rows = NULL, theme = tt)
  tbl2 <- tableGrob(stats_df_table2, rows = NULL, theme = tt)

  # Apply formatting to both tables
  tbl1 <- format_table(tbl1, stats_df_table1, pitch_names, header_fontsize = 10)
  tbl2 <- format_table(tbl2, stats_df_table2, pitch_names, header_fontsize = 10)

  # Combine both tables with some spacing
  arrangeGrob(
    tbl1,
    tbl2,
    ncol = 1,
    heights = c(1, 1),
    padding = unit(1.5, "cm")
  )
}

# ---- Main Processing Function ----
# OnePitcher mode: every game date gets its own page (per_date_pages = TRUE)
generate_pitcher_reports <- function(input_file, output_dir,
                                     pitcher_filter = NULL,
                                     start_date = NULL,
                                     end_date = NULL) {
  generate_pitcher_reports_core(
    input_file, output_dir,
    pitcher_filter = pitcher_filter,
    start_date = start_date,
    end_date = end_date,
    per_date_pages = TRUE,
    plot_fun = create_pitch_plot,
    tables_fun = create_pitcher_tables
  )
}

# ---- Execute Main Function ----
# Pass the optional filter parameters to the function
generate_pitcher_reports(
  input_file_path,
  output_directory,
  pitcher_filter = selected_pitcher_filter,
  start_date = start_date_filter,
  end_date = end_date_filter
)
