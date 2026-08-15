# ---- Configuration ----
input_file_path <- "FCL"  # Team code (e.g., "PIT", "WSH") or full path
output_directory <- path.expand("~/Downloads/")

# ---- Optional Filtering Parameters ----
# Set these to NULL to disable filtering
selected_pitcher_filter <- NULL  # Example format: "Bieber, Shane"
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

# Allow command-line overrides: Rscript Season.R "/path/to/file.csv" "2026-04-01" "2026-04-08" "Bieber, Shane"
cli_args <- commandArgs(trailingOnly = TRUE)
if (length(cli_args) >= 1) input_file_path <- cli_args[1]
if (length(cli_args) >= 2) start_date_filter <- cli_args[2]
if (length(cli_args) >= 3) end_date_filter <- cli_args[3]
if (length(cli_args) >= 4) selected_pitcher_filter <- cli_args[4]

# Resolve team code to full path (e.g., "PIT" -> "~/Downloads/NLC2026 - PIT.csv")
input_file_path <- resolve_team_path(input_file_path)

# ---- Data Processing Functions ----

# Function to calculate platoon split stats
calculate_platoon_stats <- function(data, pitcher_name) {
  pitcher_data <- data %>%
    filter(Pitcher == pitcher_name)

  # Define pitch outcome event categories (simplified descriptions)
  swing_events <- c("Swinging Strike", "Foul", "In Play")
  csw_events <- c("Called Strike", "Swinging Strike")
  swstr_events <- c("Swinging Strike")
  in_play_events <- c("In Play")

  # Calculate total pitches by handedness first
  total_pitches_by_handedness <- pitcher_data %>%
    group_by(Bats) %>%
    summarise(total_pitches = n(), .groups = "drop")

  # Calculate stats by pitch type and batter handedness
  platoon_stats <- pitcher_data %>%
    group_by(`Pitch Type`, Bats) %>%
    summarize(
      num_thrown = n(),
      # IZ% count
      iz_count = sum(InZone == "Yes", na.rm = TRUE),
      swing_count = sum(Description %in% swing_events, na.rm = TRUE),
      csw_count = sum(Description %in% csw_events, na.rm = TRUE),
      swstr_count = sum(Description %in% swstr_events, na.rm = TRUE),
      # Chase count (swings outside zone) and out-of-zone pitch count
      ooz_count = sum(InZone == "No", na.rm = TRUE),
      chase_count = sum(Description %in% swing_events & (InZone == "No"), na.rm = TRUE),
      # Ground Ball calculations
      balls_in_play = sum(Description %in% in_play_events & !grepl("^bunt", BBType), na.rm = TRUE),
      ground_balls = sum(Description %in% in_play_events & BBType == "ground_ball", na.rm = TRUE),
      .groups = "drop"
    ) %>%
    # Join with total pitches by handedness to calculate correct percentages
    left_join(total_pitches_by_handedness, by = "Bats") %>%
    mutate(
      num_thrown_fmt = sprintf("%.0f", num_thrown),
      percent_thrown = sprintf("%.1f%%", (num_thrown / total_pitches) * 100),
      iz_percent = sprintf("%.1f%%", (iz_count / num_thrown) * 100),
      csw_percent = sprintf("%.1f%%", (csw_count / num_thrown) * 100),
      swstr_percent = ifelse(swing_count > 0, sprintf("%.1f%%", (swstr_count / swing_count) * 100), "---"),
      chase_percent = ifelse(ooz_count > 0, sprintf("%.1f%%", (chase_count / ooz_count) * 100), "---"),
      gb_percent = sprintf("%.1f%%", ifelse(balls_in_play > 0, (ground_balls / balls_in_play) * 100, 0))
    ) %>%
    select(`Pitch Type`, Bats, num_thrown_fmt, percent_thrown, iz_percent,
           csw_percent, swstr_percent, chase_percent, gb_percent)

  # Pivot to create side-by-side columns for RHH and LHH
  rhh_stats <- platoon_stats %>%
    filter(Bats == "R") %>%
    select(-Bats) %>%
    rename_with(~paste0(., "_rhh"), -`Pitch Type`)

  lhh_stats <- platoon_stats %>%
    filter(Bats == "L") %>%
    select(-Bats) %>%
    rename_with(~paste0(., "_lhh"), -`Pitch Type`)

  # Combine RHH and LHH stats
  combined_platoon <- rhh_stats %>%
    full_join(lhh_stats, by = "Pitch Type") %>%
    replace_na(list(
      num_thrown_fmt_rhh = "0", percent_thrown_rhh = "0.0%", iz_percent_rhh = "0.0%",
      csw_percent_rhh = "0.0%", swstr_percent_rhh = "---",
      chase_percent_rhh = "---", gb_percent_rhh = "0.0%",
      num_thrown_fmt_lhh = "0", percent_thrown_lhh = "0.0%", iz_percent_lhh = "0.0%",
      csw_percent_lhh = "0.0%", swstr_percent_lhh = "---",
      chase_percent_lhh = "---", gb_percent_lhh = "0.0%"
    ))

  return(combined_platoon)
}

# ---- Visualization Functions ----

# Pitch movement plot: titled, with per-pitch-type arm-angle rays
create_pitch_plot <- function(pitch_data_filtered, pitcher_name, game_date = NULL) {
  create_pitch_plot_shared(pitch_data_filtered, pitcher_name, game_date,
                           arm_angle_lines = TRUE, show_title = TRUE)
}

# Function to create both main stats table and platoon splits table
create_pitcher_tables <- function(pitch_data, selected_pitcher, game_date = NULL) {
  # Filter by date if provided
  if (!is.null(game_date)) {
    pitch_data <- pitch_data %>%
      filter(`Game Date` == game_date)
  }

  # Check if Arm Angle column exists and has data for this pitcher
  has_arm_angle <- col_has_data(pitch_data, selected_pitcher, "ArmAngle")

  # FIRST TABLE - Combined stats (pitch metrics + outcome metrics)
  stats_full <- summarize_pitch_type_stats(pitch_data, selected_pitcher,
                                           has_arm_angle,
                                           gb_zero = sprintf("%.1f%%", 0))

  # Handle case where no stats are available
  if (nrow(stats_full) == 0) {
    return(grid.text("No pitch data available", gp = gpar(fontsize = 16, fontface = "bold")))
  }

  combined_cols <- c(
    "Pitch Type", "num_thrown", "percent_thrown", "avg_velo", "max_velo", "avg_spin",
    "avg_rtilt", "avg_tilt", "avg_ivb", "avg_hb", "avg_height", "avg_side",
    "avg_extension", if (has_arm_angle) "avg_arm_angle", "avg_vaa", "avg_haa",
    "iz_percent", "csw_percent", "swstr_percent", "chase_percent", "gb_percent"
  )
  stats_df <- map_and_sort_pitch_types(stats_full %>% select(all_of(combined_cols)))
  names(stats_df) <- c(
    "Pitch Type", "Count", "% Thrown", "Velocity", "Max Velo", "Spin Rate", "RTilt", "OTilt",
    "IVB", "HB", "RelZ", "RelX", "Ext", if (has_arm_angle) "Arm Angle",
    "VAA", "HAA",
    "Zone%", "CSW%", "Whiff%", "Chase%", "GB%"
  )

  # SECOND TABLE - Platoon splits
  platoon_df <- calculate_platoon_stats(pitch_data, selected_pitcher)

  if (nrow(platoon_df) > 0) {
    # Replace pitch codes with full names
    platoon_df$`Pitch Type` <- pitch_names[platoon_df$`Pitch Type`]
    platoon_df <- platoon_df[!is.na(platoon_df$`Pitch Type`), ]

    # Convert pitch_type to factor (levels act as the usage tie-break order)
    platoon_df$`Pitch Type` <- factor(platoon_df$`Pitch Type`, levels = pitch_order)

    # Sort by overall usage (RHH + LHH counts, descending) so the platoon
    # table matches the main table's order; exact ties fall back to pitch_order
    platoon_df <- platoon_df[order(
      -(as.numeric(platoon_df$num_thrown_fmt_rhh) + as.numeric(platoon_df$num_thrown_fmt_lhh)),
      platoon_df$`Pitch Type`), ]

    # Add duplicate pitch type column for LHH section
    platoon_df$pitch_type_lhh <- platoon_df$`Pitch Type`

    # Define headers for the platoon table with duplicate pitch type column
    platoon_colnames <- c(
      "Pitch Type", "Count", "% Thrown", "Zone%", "CSW%", "Whiff%", "Chase%", "GB%",
      "Pitch Type", "Count", "% Thrown", "Zone%", "CSW%", "Whiff%", "Chase%", "GB%"
    )

    # Reorder columns to include duplicate pitch type
    platoon_df <- platoon_df %>%
      select(`Pitch Type`, num_thrown_fmt_rhh, percent_thrown_rhh, iz_percent_rhh,
             csw_percent_rhh, swstr_percent_rhh, chase_percent_rhh, gb_percent_rhh,
             pitch_type_lhh, num_thrown_fmt_lhh, percent_thrown_lhh, iz_percent_lhh,
             csw_percent_lhh, swstr_percent_lhh, chase_percent_lhh, gb_percent_lhh)

    # Set column names
    names(platoon_df) <- platoon_colnames
  }

  # Create base table theme (tight horizontal padding so each column is only
  # as wide as its widest content)
  tt <- make_table_theme(core_fontsize = 9, head_fontsize = 9, pad_h_mm = 7)

  # Create the main stats table (column widths auto-size to fit content)
  tbl1 <- tableGrob(stats_df, rows = NULL, theme = tt)

  # Apply formatting to the main table
  tbl1 <- format_table(tbl1, stats_df, pitch_names, header_fontsize = 9,
                       color_platoon_cols = TRUE)

  # Create platoon table if data exists
  if (nrow(platoon_df) > 0) {
    tbl2 <- tableGrob(platoon_df, rows = NULL, theme = tt)

    # Apply formatting to the platoon table
    tbl2 <- format_table(tbl2, platoon_df, pitch_names, header_fontsize = 9,
                         color_platoon_cols = TRUE)

    # Add headers for vs RHH and vs LHH
    header_grob <- textGrob(
      c("vs RHH", "vs LHH"),
      x = c(0.3, 0.7),
      y = 0.5,
      gp = gpar(fontface = "bold", fontsize = 10)
    )

    # Combine both tables
    combined_tables <- arrangeGrob(
      tbl1,
      header_grob,
      tbl2,
      ncol = 1,
      heights = c(4, 0.3, 2)
    )
  } else {
    combined_tables <- tbl1
  }

  return(combined_tables)
}

# ---- Main Processing Function ----
# Season mode: one combined page per pitcher (per_date_pages = FALSE)
generate_pitcher_reports <- function(input_file, output_dir,
                                     pitcher_filter = NULL,
                                     start_date = NULL,
                                     end_date = NULL) {
  generate_pitcher_reports_core(
    input_file, output_dir,
    pitcher_filter = pitcher_filter,
    start_date = start_date,
    end_date = end_date,
    per_date_pages = FALSE,
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
