# ---- Configuration ----
input_file_path <- "" # Team code (e.g., "PIT", "WSH") or full path
output_directory <- path.expand("~/Downloads/")
# ---- Optional Filtering Parameters ----
# Set these to NULL to disable filtering
selected_pitcher_filter <- NULL          # Example format: "Bieber, Shane" - Set to NULL for all pitchers
start_date_filter <- "2026-08-17"        # Example format: "2025-05-18" - Set to NULL for no date filter
end_date_filter <- "2026-08-17"          # Example format: "2025-05-18" - Set to NULL for no date filter
# ---- Required Libraries ----
library(tidyverse)
library(patchwork)
library(gridExtra)
library(gtable)
library(grid)
library(cowplot)
library(jsonlite)

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

# Allow command-line overrides: Rscript Daily.R "/path/to/file.csv" "2026-04-08" "2026-04-08" "Bieber, Shane"
cli_args <- commandArgs(trailingOnly = TRUE)
if (length(cli_args) >= 1)
  input_file_path <- cli_args[1]
if (length(cli_args) >= 2)
  start_date_filter <- cli_args[2]
if (length(cli_args) >= 3)
  end_date_filter <- cli_args[3]
if (length(cli_args) >= 4)
  selected_pitcher_filter <- cli_args[4]

# Resolve team code to full path (e.g., "PIT" -> "~/Downloads/NLC2026 - PIT.csv")
input_file_path <- resolve_team_path(input_file_path)

# ---- MLB Team ID Mapping ----
mlb_team_ids <- c(
  "ARI" = 109,
  "ATL" = 144,
  "BAL" = 110,
  "BOS" = 111,
  "CHC" = 112,
  "CWS" = 145,
  "CIN" = 113,
  "CLE" = 114,
  "COL" = 115,
  "DET" = 116,
  "HOU" = 117,
  "KCR" = 118,
  "LAA" = 108,
  "LAD" = 119,
  "MIA" = 146,
  "MIL" = 158,
  "MIN" = 142,
  "NYM" = 121,
  "NYY" = 147,
  "ATH" = 133,
  "PHI" = 143,
  "PIT" = 134,
  "SDP" = 135,
  "SFG" = 137,
  "SEA" = 136,
  "STL" = 138,
  "TBR" = 139,
  "TEX" = 140,
  "TOR" = 141,
  "WSH" = 120
)

# ---- Data Processing Functions ----

# Function to calculate pitch stats for combined table.
# Daily-specific: every pitch-metric column is conditional on the CSV actually
# having data for it (same-day scrapes often lack the Savant supplement), a
# Stuff+ column appears when the Python scorer added one, and a Total row is
# appended. Deliberately NOT merged with the shared summarize_pitch_type_stats.
calculate_pitcher_stats <- function(data, pitcher_name) {
  pitcher_data <- data %>%
    filter(Pitcher == pitcher_name)

  total_pitches <- nrow(pitcher_data)

  # Define pitch outcome event categories
  swing_events <- c("Swinging Strike", "Foul", "In Play")
  csw_events <- c("Called Strike", "Swinging Strike")
  swstr_events <- c("Swinging Strike")
  in_play_events <- c("In Play")

  # --- Determine which conditional pitch metric columns have data ---
  conditional_cols <- c(
    "Velocity",
    "Spin Rate",
    "RTilt",
    "OTilt",
    "xIndVrtBrk",
    "xHorzBrk",
    "RelPosZ",
    "RelPosX",
    "Extension",
    "ArmAngle",
    "VAA",
    "HAA"
  )
  has_col <- setNames(sapply(conditional_cols, function(col)
    col_has_data(data, pitcher_name, col)),
    conditional_cols)

  # Check for Stuff+ column (added by Python scorer)
  has_stuff_plus <- col_has_data(data, pitcher_name, "Stuff+")

  # Check for BB Type (needed for GB%)
  has_bb_type <- "BBType" %in% names(data)

  # --- Build the stats summary ---
  result <- pitcher_data %>%
    group_by(`Pitch Type`) %>%
    summarize(
      # Always-present columns
      num_thrown = sprintf("%.0f", n()),
      percent_thrown = sprintf("%.1f%%", n() / total_pitches * 100),

      # Conditional pitch metric columns
      avg_velo = if (has_col["Velocity"])
        sprintf("%.1f mph", mean(Velocity, na.rm = TRUE))
      else
        NA_character_,
      max_velo = if (has_col["Velocity"])
        sprintf("%.1f mph", max(Velocity, na.rm = TRUE))
      else
        NA_character_,
      avg_spin = if (has_col["Spin Rate"])
        sprintf("%.0f rpm", round(mean(
          `Spin Rate`, na.rm = TRUE
        )))
      else
        NA_character_,
      avg_rtilt = if (has_col["RTilt"])
        avg_tilt_clock(`RTilt`)
      else
        NA_character_,
      avg_tilt = if (has_col["OTilt"])
        avg_tilt_clock(`OTilt`)
      else
        NA_character_,
      avg_ivb = if (has_col["xIndVrtBrk"])
        sprintf("%.1f\"", mean(xIndVrtBrk, na.rm = TRUE))
      else
        NA_character_,
      avg_hb = if (has_col["xHorzBrk"])
        sprintf("%.1f\"", mean(xHorzBrk, na.rm = TRUE))
      else
        NA_character_,
      avg_height = if (has_col["RelPosZ"])
        sprintf("%.2f'", mean(RelPosZ, na.rm = TRUE))
      else
        NA_character_,
      avg_side = if (has_col["RelPosX"])
        sprintf("%.2f'", mean(RelPosX, na.rm = TRUE))
      else
        NA_character_,
      avg_extension = if (has_col["Extension"])
        sprintf("%.2f'", mean(Extension, na.rm = TRUE))
      else
        NA_character_,
      avg_arm_angle = if (has_col["ArmAngle"])
        sprintf("%.1f", mean(ArmAngle, na.rm = TRUE))
      else
        NA_character_,
      avg_vaa = if (has_col["VAA"])
        sprintf("%.2f", mean(VAA, na.rm = TRUE))
      else
        NA_character_,
      avg_haa = if (has_col["HAA"])
        sprintf("%.2f", mean(HAA, na.rm = TRUE))
      else
        NA_character_,

      # Stuff+ (conditional on column existing in CSV)
      avg_stuff_plus = if (has_stuff_plus)
        sprintf("%.0f", round(mean(`Stuff+`, na.rm = TRUE)))
      else
        NA_character_,

      # Always-present outcome columns
      iz_percent = sprintf("%.1f%%", sum(InZone == "Yes", na.rm = TRUE) / n() * 100),
      csw_percent = sprintf(
        "%.1f%%",
        sum(Description %in% csw_events, na.rm = TRUE) / n() * 100
      ),
      swstr_percent = {
        total_swings <- sum(Description %in% swing_events, na.rm = TRUE)
        if (total_swings > 0) {
          sprintf("%.1f%%",
                  sum(Description %in% swstr_events, na.rm = TRUE) / total_swings * 100)
        } else {
          "---"
        }
      },
      chase_percent = {
        ooz_pitches <- sum(InZone == "No", na.rm = TRUE)
        if (ooz_pitches > 0) {
          ooz_swings <- sum(Description %in% swing_events &
                              (InZone == "No"), na.rm = TRUE)
          sprintf("%.1f%%", ooz_swings / ooz_pitches * 100)
        } else {
          "---"
        }
      },
      # GB% (always shown, uses BB Type if available)
      gb_percent = {
        if (has_bb_type) {
          total_bip <- sum(Description %in% in_play_events &
                             !grepl("^bunt", BBType),
                           na.rm = TRUE)
          if (total_bip > 0) {
            # Gate on the in-play description too, matching the total_bip
            # denominator (see summarize_pitch_type_stats in
            # pitcher_report_utils.R) — otherwise a ground_ball tagged on a
            # non "In Play" row inflates GB% and can exceed 100%.
            n_gb <- sum(Description %in% in_play_events &
                          BBType == "ground_ball", na.rm = TRUE)
            sprintf("%.1f%%", n_gb / total_bip * 100)
          } else {
            "---"
          }
        } else {
          "---"
        }
      },
      .groups = "drop"
    )

  # --- Compute and append Total row ---
  total_swings_all <- sum(pitcher_data$Description %in% swing_events, na.rm = TRUE)
  total_bip_all <- sum(
    pitcher_data$Description %in% in_play_events &
      !grepl("^bunt", pitcher_data$BBType),
    na.rm = TRUE
  )

  total_row <- tibble(
    `Pitch Type` = "Total",
    num_thrown = sprintf("%.0f", total_pitches),
    percent_thrown = NA_character_,
    avg_velo = NA_character_,
    max_velo = NA_character_,
    avg_spin = NA_character_,
    avg_rtilt = NA_character_,
    avg_tilt = NA_character_,
    avg_ivb = NA_character_,
    avg_hb = NA_character_,
    avg_height = NA_character_,
    avg_side = NA_character_,
    avg_extension = NA_character_,
    avg_arm_angle = NA_character_,
    avg_vaa = NA_character_,
    avg_haa = NA_character_,
    avg_stuff_plus = NA_character_,
    iz_percent = sprintf(
      "%.1f%%",
      sum(pitcher_data$InZone == "Yes", na.rm = TRUE) / total_pitches * 100
    ),
    csw_percent = sprintf(
      "%.1f%%",
      sum(pitcher_data$Description %in% csw_events, na.rm = TRUE) / total_pitches * 100
    ),
    swstr_percent = if (total_swings_all > 0)
      sprintf(
        "%.1f%%",
        sum(pitcher_data$Description %in% swstr_events, na.rm = TRUE) / total_swings_all * 100
      )
    else
      "---",
    chase_percent = {
      ooz_all <- sum(pitcher_data$InZone == "No", na.rm = TRUE)
      if (ooz_all > 0)
        sprintf(
          "%.1f%%",
          sum(
            pitcher_data$Description %in% swing_events &
              (pitcher_data$InZone == "No"),
            na.rm = TRUE
          ) / ooz_all * 100
        )
      else
        "---"
    },
    gb_percent = if (has_bb_type &&
                     total_bip_all > 0)
      sprintf(
        "%.1f%%",
        sum(
          pitcher_data$Description %in% in_play_events &
            pitcher_data$BBType == "ground_ball",
          na.rm = TRUE
        ) / total_bip_all * 100
      )
    else
      "---"
  )

  result <- bind_rows(result, total_row)

  # Replace NA with empty string so Total row shows blank for pitch metrics
  result[is.na(result)] <- ""

  # --- Build final column selection, dropping any columns that are all NA ---
  # Start with always-present columns
  cols_to_keep <- c("Pitch Type", "num_thrown", "percent_thrown")

  # Add Stuff+ if present
  if (has_stuff_plus)
    cols_to_keep <- c(cols_to_keep, "avg_stuff_plus")

  # Add conditional pitch metric columns (only if they have data)
  if (has_col["Velocity"])
    cols_to_keep <- c(cols_to_keep, "avg_velo", "max_velo")
  if (has_col["Spin Rate"])
    cols_to_keep <- c(cols_to_keep, "avg_spin")
  if (has_col["RTilt"])
    cols_to_keep <- c(cols_to_keep, "avg_rtilt")
  if (has_col["OTilt"])
    cols_to_keep <- c(cols_to_keep, "avg_tilt")
  if (has_col["xIndVrtBrk"])
    cols_to_keep <- c(cols_to_keep, "avg_ivb")
  if (has_col["xHorzBrk"])
    cols_to_keep <- c(cols_to_keep, "avg_hb")
  if (has_col["RelPosZ"])
    cols_to_keep <- c(cols_to_keep, "avg_height")
  if (has_col["RelPosX"])
    cols_to_keep <- c(cols_to_keep, "avg_side")
  if (has_col["Extension"])
    cols_to_keep <- c(cols_to_keep, "avg_extension")
  if (has_col["ArmAngle"])
    cols_to_keep <- c(cols_to_keep, "avg_arm_angle")
  if (has_col["VAA"])
    cols_to_keep <- c(cols_to_keep, "avg_vaa")
  if (has_col["HAA"])
    cols_to_keep <- c(cols_to_keep, "avg_haa")

  # Outcome columns. CSW% and Whiff% derive from Description and always show.
  # Zone%, Chase% and GB% drop when InZone / BBType are empty for the whole
  # outing, where they would otherwise print a false "0.0%" rather than blank.
  cols_to_keep <- c(
    cols_to_keep,
    keep_populated_cols(
      c("iz_percent", "csw_percent", "swstr_percent", "chase_percent", "gb_percent"),
      data,
      pitcher_name
    )
  )

  result <- result %>% select(all_of(cols_to_keep))

  # --- Assign display header names ---
  col_name_map <- c(
    "Pitch Type" = "Pitch Type",
    "num_thrown" = "Count",
    "percent_thrown" = "% Thrown",
    "avg_stuff_plus" = "Stuff+",
    "avg_velo" = "Velo",
    "max_velo" = "Max Velo",
    "avg_spin" = "Spin Rate",
    "avg_rtilt" = "RTilt",
    "avg_tilt" = "OTilt",
    "avg_ivb" = "IVB",
    "avg_hb" = "HB",
    "avg_height" = "RelZ",
    "avg_side" = "RelX",
    "avg_extension" = "Ext",
    "avg_arm_angle" = "Arm Angle",
    "avg_vaa" = "VAA",
    "avg_haa" = "HAA",
    "iz_percent" = "Zone%",
    "csw_percent" = "CSW%",
    "swstr_percent" = "Whiff%",
    "chase_percent" = "Chase%",
    "gb_percent" = "GB%"
  )

  names(result) <- col_name_map[names(result)]

  return(result)
}

# ---- Visualization Functions ----

# Pitch movement plot: untitled (Daily uses a separate title grob), with
# per-pitch-type arm-angle rays
create_pitch_plot <- function(pitch_data_filtered, pitcher_name, game_date = NULL) {
  create_pitch_plot_shared(pitch_data_filtered, pitcher_name, game_date,
                           arm_angle_lines = TRUE, show_title = FALSE)
}

# Function to create pitcher stats table
create_pitcher_tables <- function(pitch_data, selected_pitcher, game_date = NULL) {
  # Filter by date if provided
  if (!is.null(game_date)) {
    pitch_data <- pitch_data %>%
      filter(`Game Date` == game_date)
  }

  stats_df <- calculate_pitcher_stats(pitch_data, selected_pitcher)

  # Handle case where no stats are available
  if (nrow(stats_df) == 0) {
    return(grid.text(
      "No pitch data available",
      gp = gpar(fontsize = 16, fontface = "bold")
    ))
  }

  # Replace pitch codes with full names (preserve "Total" as-is)
  stats_df$`Pitch Type` <- ifelse(stats_df$`Pitch Type` == "Total", "Total", pitch_names[stats_df$`Pitch Type`])

  # Drop rows whose code has no mapped name (unmapped pitch types). These never
  # occur in practice; if one slipped through, an NA Pitch Type would crash the
  # Total-row check in format_table and inject a phantom all-NA row via NA
  # logical subsetting in the sort below.
  stats_df <- stats_df[!is.na(stats_df$`Pitch Type`), ]

  # Sort by usage (descending), Total always last; exact usage ties fall
  # back to pitch_order
  total_mask <- stats_df$`Pitch Type` == "Total"
  non_total <- stats_df[!total_mask, ]
  total_rows <- stats_df[total_mask, ]
  non_total <- non_total[order(
    -as.numeric(non_total$Count),
    factor(non_total$`Pitch Type`, levels = pitch_order)
  ), ]
  stats_df <- bind_rows(non_total, total_rows)

  # Create base table theme (tight horizontal padding so each column is only
  # as wide as its widest content; colhead fontsize matches the format_table
  # header override so fitted widths don't clip the bold headers)
  tt <- make_table_theme(core_fontsize = 11, head_fontsize = 12, pad_h_mm = 7)

  # Create the table (auto-size column widths to fit content)
  tbl <- tableGrob(stats_df, rows = NULL, theme = tt)

  # Apply formatting to the table
  tbl <- format_table(tbl, stats_df, pitch_names, header_fontsize = 12)

  return(tbl)
}

# ---- MLB Stats API Functions ----
# Fetch pitcher game statline (IP, H, R, ER, SO, BB) from MLB Stats API
fetch_pitcher_statline <- function(game_date, team_code, pitcher_name) {
  tryCatch({
    # Look up team ID
    team_id <- mlb_team_ids[team_code]
    if (is.na(team_id)) {
      message("Unknown team code: ", team_code)
      return(NULL)
    }

    # Format date for API (MM/DD/YYYY)
    api_date <- format(as.Date(game_date), "%m/%d/%Y")

    # Get schedule for this team on this date
    schedule_url <- paste0(
      "https://statsapi.mlb.com/api/v1/schedule?date=",
      api_date,
      "&sportId=1&teamId=",
      team_id
    )
    schedule <- fromJSON(schedule_url)

    if (length(schedule$dates) == 0)
      return(NULL)

    games <- schedule$dates$games[[1]]
    if (is.null(games) || nrow(games) == 0)
      return(NULL)

    # Convert "Last, First" to "First Last" for matching
    name_parts <- str_match(pitcher_name, "^(.+),\\s*(.+)$")
    if (!is.na(name_parts[1, 1])) {
      search_name <- tolower(paste(trimws(name_parts[1, 3]), trimws(name_parts[1, 2])))
    } else {
      search_name <- tolower(pitcher_name)
    }

    # Search one game's boxscore for the pitcher; NULL when he isn't found
    # with a non-empty pitching block.
    search_game <- function(game_pk) {
      # Get boxscore
      box_url <- paste0("https://statsapi.mlb.com/api/v1/game/",
                        game_pk,
                        "/boxscore")
      boxscore <- fromJSON(box_url)

      # Search both sides for the pitcher
      for (side in c("away", "home")) {
        players <- boxscore$teams[[side]]$players
        if (is.null(players))
          next

        for (pkey in names(players)) {
          p <- players[[pkey]]
          full_name <- p$person$fullName
          if (is.null(full_name))
            next

          if (tolower(full_name) == search_name) {
            ps <- p$stats$pitching
            # Empty pitching block: rostered but didn't throw in THIS game
            # (e.g. the other half of a doubleheader) — keep scanning
            if (is.null(ps) || length(ps) == 0)
              next

            return(
              list(
                ip = if (!is.null(ps$inningsPitched))
                  ps$inningsPitched
                else
                  "---",
                h  = if (!is.null(ps$hits))
                  as.character(ps$hits)
                else
                  "---",
                r  = if (!is.null(ps$runs))
                  as.character(ps$runs)
                else
                  "---",
                er = if (!is.null(ps$earnedRuns))
                  as.character(ps$earnedRuns)
                else
                  "---",
                so = if (!is.null(ps$strikeOuts))
                  as.character(ps$strikeOuts)
                else
                  "---",
                bb = if (!is.null(ps$baseOnBalls))
                  as.character(ps$baseOnBalls)
                else
                  "---"
              )
            )
          }
        }
      }
      NULL
    }

    # Scan ALL games for this team/date (doubleheaders have two gamePks, and a
    # pitcher can appear with an empty pitching block in the game he didn't
    # throw in); return the first non-empty statline.
    for (game_pk in games$gamePk) {
      statline <- search_game(game_pk)
      if (!is.null(statline))
        return(statline)
    }

    message("Pitcher '",
            pitcher_name,
            "' not found with a pitching statline in any boxscore on ",
            game_date)
    return(NULL)
  }, error = function(e) {
    message("MLB API error: ", e$message)
    return(NULL)
  })
}

# Create a title grob: "First Last (M-DD-YYYY)"
create_title_grob <- function(pitcher_name, game_date = NULL) {
  pitcher_name_fmt <- str_replace(pitcher_name, "(.*), (.*)", "\\2 \\1")

  if (!is.null(game_date)) {
    # Format date as M-DD-YYYY (no leading zero on month)
    date_obj <- as.Date(game_date)
    date_str <- paste0(as.integer(format(date_obj, "%m")),
                       "-",
                       format(date_obj, "%d"),
                       "-",
                       format(date_obj, "%Y"))
    title_text <- paste0(pitcher_name_fmt, " (", date_str, ")")
  } else {
    title_text <- pitcher_name_fmt
  }

  textGrob(title_text, gp = gpar(fontsize = 24, fontface = "bold"))
}

# Create a clean statline table (similar style to pitch metrics)
create_statline_table <- function(statline) {
  if (is.null(statline))
    return(NULL)

  stat_df <- data.frame(
    IP = statline$ip,
    H  = statline$h,
    R  = statline$r,
    ER = statline$er,
    SO = statline$so,
    BB = statline$bb,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  tt <- ttheme_minimal(
    core = list(
      fg_params = list(col = "black", fontsize = 13),
      bg_params = list(fill = "white"),
      padding = unit(c(10, 7), "mm")
    ),
    colhead = list(
      fg_params = list(
        col = "black",
        fontface = "bold",
        fontsize = 13
      ),
      bg_params = list(fill = "gray90"),
      padding = unit(c(10, 7), "mm")
    )
  )

  tbl <- tableGrob(stat_df, rows = NULL, theme = tt)

  # Add border around the table
  tbl <- gtable::gtable_add_grob(
    tbl,
    grobs = rectGrob(gp = gpar(fill = NA, col = "black")),
    t = 1,
    b = nrow(tbl),
    l = 1,
    r = ncol(tbl)
  )

  # Add row borders
  for (i in 1:nrow(tbl)) {
    tbl <- gtable::gtable_add_grob(
      tbl,
      grobs = rectGrob(gp = gpar(
        fill = NA, col = "black"
      )),
      t = i,
      b = i,
      l = 1,
      r = ncol(tbl)
    )
  }

  # Add column borders
  for (j in 1:ncol(tbl)) {
    tbl <- gtable::gtable_add_grob(
      tbl,
      grobs = rectGrob(gp = gpar(
        fill = NA, col = "black"
      )),
      t = 1,
      b = nrow(tbl),
      l = j,
      r = j
    )
  }

  return(tbl)
}

# ---- Page Assembly ----

# Compose one report page: title -> [statline] -> plot -> metrics table
daily_page_grid <- function(title_grob, statline_table, pitch_plot, table_plot) {
  if (!is.null(statline_table)) {
    plot_grid(
      title_grob,
      nullGrob(),
      statline_table,
      nullGrob(),
      pitch_plot,
      nullGrob(),
      table_plot,
      ncol = 1,
      rel_heights = c(0.08, 0.04, 0.1, 0.03, 1.2, 0.03, 1)
    )
  } else {
    plot_grid(
      title_grob,
      nullGrob(),
      pitch_plot,
      nullGrob(),
      table_plot,
      ncol = 1,
      rel_heights = c(0.08, 0.03, 1.2, 0.03, 1)
    )
  }
}

# Render the PDF page(s) for one pitcher: title grob + MLB API statline +
# movement plot + stats table, one page per game date.
render_daily_pitcher_pages <- function(pitcher_data, pitcher_name, team_filter) {
  # Get unique game dates for this pitcher
  game_dates <- unique(pitcher_data$`Game Date`)

  # If there's only one date, process as before
  if (length(game_dates) == 1) {
    # Skip pitchers with no valid movement data
    pitch_data_filtered <- pitcher_data %>%
      drop_na(xHorzBrk, xIndVrtBrk)

    if (nrow(pitch_data_filtered) == 0) {
      message(paste("Skipping", pitcher_name, "- no valid movement data"))
      return(invisible(FALSE))
    }

    # Create title grob
    title_grob <- create_title_grob(pitcher_name, game_dates[1])

    # Fetch statline from MLB Stats API
    statline <- fetch_pitcher_statline(game_dates[1], team_filter, pitcher_name)
    statline_table <- create_statline_table(statline)

    # Create the pitch movement plot
    pitch_plot <- create_pitch_plot(pitch_data_filtered, pitcher_name)

    # Create the stats table
    table_plot <- create_pitcher_tables(pitcher_data, pitcher_name)

    # Combine: title -> statline -> space -> plot -> space -> metrics,
    # print to the current PDF page
    print(daily_page_grid(title_grob, statline_table, pitch_plot, table_plot))
    message(paste("Created plot for", pitcher_name))
  } else {
    # If there are multiple dates, create a separate page for each date
    message(paste(
      "Found",
      length(game_dates),
      "different game dates for",
      pitcher_name
    ))

    # Sort game dates chronologically (earliest to latest)
    game_dates <- as.character(sort(game_dates))

    for (game_date in game_dates) {
      # Filter data for this specific date
      date_data <- pitcher_data %>%
        filter(`Game Date` == game_date)

      # Filter for movement data
      pitch_data_filtered <- date_data %>%
        drop_na(xHorzBrk, xIndVrtBrk)

      if (nrow(pitch_data_filtered) == 0) {
        message(
          paste(
            "Skipping",
            pitcher_name,
            "on",
            game_date,
            "- no valid movement data"
          )
        )
        next
      }

      # Create title grob
      date_string <- as.character(game_date)
      title_grob <- create_title_grob(pitcher_name, date_string)

      # Fetch statline from MLB Stats API
      statline <- fetch_pitcher_statline(game_date, team_filter, pitcher_name)
      statline_table <- create_statline_table(statline)

      # Create the pitch movement plot
      pitch_plot <- create_pitch_plot(pitch_data_filtered, pitcher_name, date_string)

      # Create the stats table for this date
      table_plot <- create_pitcher_tables(date_data, pitcher_name)

      # Combine: title -> statline -> space -> plot -> space -> metrics,
      # print to the current PDF page
      print(daily_page_grid(title_grob, statline_table, pitch_plot, table_plot))
      message(paste("Created plot for", pitcher_name, "on", game_date))
    }
  }
  invisible(TRUE)
}

# ---- Main Processing Function ----
generate_pitcher_reports <- function(input_file,
                                     output_dir,
                                     pitcher_filter = NULL,
                                     start_date = NULL,
                                     end_date = NULL) {
  # Read data; RTilt/OTilt stay character to prevent time parsing — handled
  # inside load_pitch_data().
  pitch_data <- read_team_pitch_data(input_file)

  # Extract team from filename
  filename <- basename(input_file)
  # Updated pattern to match "2025 Playoffs - CLE.csv" format
  team_match <- str_extract(filename, "(?<=- )[A-Z]{3}(?=\\.csv)")

  if (is.na(team_match)) {
    team_match <- str_extract(filename, "^[A-Z]{3}(?=\\.csv)")
  }
  if (is.na(team_match)) {
    stop(
      "Could not extract team code from filename. Expected format: 'ARI.csv' or '... - ARI.csv'"
    )
  }

  team_filter <- team_match
  message("Extracted team filter from filename: ", team_filter)

  # Apply team filtering
  message("Filtering for team: ", team_filter)
  original_count <- nrow(pitch_data)
  pitch_data <- pitch_data %>% filter(PTeam == team_filter)
  message(
    "Team filtering: ",
    original_count - nrow(pitch_data),
    " rows removed, ",
    nrow(pitch_data),
    " rows remaining"
  )

  # Apply date filtering if specified (with row-count logging)
  pitch_data <- filter_by_dates(pitch_data, start_date, end_date,
                                count_messages = TRUE)

  # Generate filename suffix based on filters
  filter_suffix <- ""

  # Add team info to suffix
  filter_suffix <- paste0(filter_suffix, "_", team_filter)

  # Add date info to suffix if filtered
  if (!is.null(start_date) || !is.null(end_date)) {
    if (!is.null(start_date) &&
        !is.null(end_date) && start_date == end_date) {
      date_suffix <- format(as.Date(start_date), "_%Y%m%d")
    } else {
      date_range <- paste0(ifelse(
        is.null(start_date),
        "Start",
        format(as.Date(start_date), "%Y%m%d")
      ),
      "_to_",
      ifelse(is.null(end_date), "End", format(as.Date(end_date), "%Y%m%d")))
      date_suffix <- paste0("_", date_range)
    }
    filter_suffix <- paste0(filter_suffix, date_suffix)
  }

  # Handle pitcher filtering
  if (!is.null(pitcher_filter)) {
    # If filtering for a specific pitcher, create a single report just for that pitcher
    message("Creating report for specific pitcher: ", pitcher_filter)

    # Filter for a single pitcher
    pitcher_data <- pitch_data %>% filter(Pitcher == pitcher_filter)

    if (nrow(pitcher_data) == 0) {
      message("No data found for pitcher: ",
              pitcher_filter,
              " with the current filters")
      return(NULL)
    }

    # Get the team for the pitcher
    current_team <- unique(pitcher_data$PTeam)[1]

    # Create a cleaner version of the pitcher name for the filename
    clean_pitcher_name <- clean_pitcher_filename(pitcher_filter)

    # Create pitcher-specific PDF
    # Build filename: "Last_First BAL 2-20-2026 Pitcher Report.pdf"
    if (!is.null(start_date) &&
        !is.null(end_date) && start_date == end_date) {
      date_obj <- as.Date(start_date)
      date_str <- paste0(as.integer(format(date_obj, "%m")),
                         "-",
                         format(date_obj, "%d"),
                         "-",
                         format(date_obj, "%Y"))
      pdf_filename <- paste0(
        output_dir,
        clean_pitcher_name,
        " ",
        team_filter,
        " ",
        date_str,
        " Pitcher Report.pdf"
      )
    } else {
      pdf_filename <- paste0(output_dir,
                             clean_pitcher_name,
                             filter_suffix,
                             " Pitcher Report.pdf")
    }
    message("Creating PDF: ", pdf_filename)
    pdf(pdf_filename, width = 15, height = 18)

    render_daily_pitcher_pages(pitcher_data, pitcher_filter, team_filter)

    # Close the PDF
    dev.off()
    message(paste(
      "Pitcher report for",
      pitcher_filter,
      "saved to",
      pdf_filename
    ))

  } else {
    # Original functionality - create team-based reports
    # Get list of all unique teams in the filtered data
    all_teams <- unique(pitch_data$PTeam)
    message("Found ", length(all_teams), " teams in the filtered dataset")

    # Process each team separately
    for (current_team in all_teams) {
      # Create team-specific PDF
      # Build filename: "BAL 2-20-2026 Pitcher Reports.pdf"
      if (!is.null(start_date) &&
          !is.null(end_date) && start_date == end_date) {
        date_obj <- as.Date(start_date)
        date_str <- paste0(
          as.integer(format(date_obj, "%m")),
          "-",
          format(date_obj, "%d"),
          "-",
          format(date_obj, "%Y")
        )
        pdf_filename <- paste0(output_dir,
                               current_team,
                               " ",
                               date_str,
                               " Pitcher Reports.pdf")
      } else if (!is.null(start_date) || !is.null(end_date)) {
        pdf_filename <- paste0(output_dir,
                               current_team,
                               filter_suffix,
                               " Pitcher Reports.pdf")
      } else {
        pdf_filename <- paste0(output_dir, current_team, " Pitcher Reports.pdf")
      }
      message("Creating report for team: ", current_team)
      message("PDF will be saved to: ", pdf_filename)

      # Create PDF
      pdf(pdf_filename, width = 15, height = 18)

      # Get pitchers for this team
      team_data <- pitch_data %>% filter(PTeam == current_team)
      all_pitchers <- sort(unique(team_data$Pitcher))
      message("Processing ",
              length(all_pitchers),
              " pitchers for ",
              current_team)

      # Loop through each pitcher
      for (selected_pitcher in all_pitchers) {
        # Get all data for this pitcher
        pitcher_data <- team_data %>% filter(Pitcher == selected_pitcher)

        render_daily_pitcher_pages(pitcher_data, selected_pitcher, team_filter)
      }

      # Close the PDF device for this team
      dev.off()
      message(paste(
        "Pitcher report for team",
        current_team,
        "saved to",
        pdf_filename
      ))
    }

    message("All team pitcher reports have been created")
  }
}
# ---- Execute Main Function ----
# Pass the filter parameters to the function
generate_pitcher_reports(
  input_file_path,
  output_directory,
  pitcher_filter = selected_pitcher_filter,
  start_date = start_date_filter,
  end_date = end_date_filter
)
