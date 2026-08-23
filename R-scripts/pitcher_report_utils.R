# pitcher_report_utils.R — Shared utilities for OnePitcher.R, Season.R, and Daily.R
# Sourced by each script to avoid duplication.

# ---- Pitch Type Definitions ----

# Mapping from pitch-type codes to display names
pitch_names <- c(
  "FF" = "Fastball",
  "SI" = "Sinker",
  "CU" = "Curveball",
  "SV" = "Slurve",
  "SL" = "Slider",
  "ST" = "Sweeper",
  "CH" = "Changeup",
  "FS" = "Splitter",
  "FC" = "Cutter",
  "KN" = "Knuckleball",
  "EP" = "Eephus"
)

# Colors for pitch-type visualizations
pitch_colors <- c(
  "FF" = "#0086D2",
  "SI" = "#FFB500",
  "FC" = "#A45B16",
  "CU" = "#230AA0",
  "SL" = "#FB6E00",
  "SV" = "#A00A55",
  "ST" = "#35B6FF",
  "CH" = "#00BA87",
  "FS" = "#F076BA",
  "KN" = "black",
  "EP" = "gray50"
)

# Display order for pitch types (must cover every pitch_names value —
# a missing entry becomes factor NA and sorts unpredictably on usage ties)
pitch_order <- c(
  "Fastball", "Sinker", "Cutter",
  "Slider", "Sweeper", "Curveball", "Slurve",
  "Changeup", "Splitter", "Knuckleball", "Eephus"
)

# ---- Team / League Lookup ----

TEAM_LEAGUE <- c(
  ATH = "AL", BAL = "AL", BOS = "AL", CLE = "AL", CWS = "AL",
  DET = "AL", HOU = "AL", KCR = "AL", LAA = "AL", MIN = "AL",
  NYY = "AL", SEA = "AL", TBR = "AL", TEX = "AL", TOR = "AL",
  ARI = "NL", ATL = "NL", CHC = "NL", CIN = "NL", COL = "NL",
  LAD = "NL", MIA = "NL", MIL = "NL", NYM = "NL", PHI = "NL",
  PIT = "NL", SDP = "NL", SFG = "NL", STL = "NL", WSH = "NL",
  ROC = "NL", AAA = "NL", FCL = "NL"
)

# 2026 layout: each team's data lives in its division workbook. The CSV you
# export from Google ("download as CSV") is named "<workbook> - <tab>.csv".
TEAM_WORKBOOK <- c(
  BAL="ALE2026", BOS="ALE2026", NYY="ALE2026", TBR="ALE2026", TOR="ALE2026",
  CLE="ALC2026", CWS="ALC2026", DET="ALC2026", KCR="ALC2026", MIN="ALC2026",
  ATH="ALW2026", HOU="ALW2026", LAA="ALW2026", SEA="ALW2026", TEX="ALW2026",
  ATL="NLE2026", MIA="NLE2026", NYM="NLE2026", PHI="NLE2026", WSH="NLE2026",
  ROC="NLE2026", AAA="NLE2026", FCL="NLE2026",
  CHC="NLC2026", CIN="NLC2026", MIL="NLC2026", PIT="NLC2026", STL="NLC2026",
  ARI="NLW2026", COL="NLW2026", LAD="NLW2026", SDP="NLW2026", SFG="NLW2026"
)

# Build the CSV path from a team code, using its division workbook.
# Usage: resolve_team_path("PIT")  =>  "~/Downloads/NLC2026 - PIT.csv" (expanded)
#        resolve_team_path("/Users/.../NLC2026 - PIT.csv")  =>  unchanged
resolve_team_path <- function(input, base_dir = path.expand("~/Downloads/")) {
  if (grepl("/", input) || grepl("\\.csv$", input, ignore.case = TRUE)) {
    return(input)
  }
  team <- toupper(trimws(input))
  wb <- TEAM_WORKBOOK[team]
  if (is.na(wb)) stop("Unknown team code: ", team, ". Expected one of: ", paste(names(TEAM_WORKBOOK), collapse = ", "))
  paste0(base_dir, wb, " - ", team, ".csv")
}

# Extract a team code from a bare code ("PIT") or a CSV path
# (".../NL 2026 - PIT.csv" -> "PIT"). Returns NA if none can be derived.
extract_team_code <- function(input) {
  s <- trimws(input)
  if (grepl("\\.csv$", s, ignore.case = TRUE)) {
    base <- sub("\\.csv$", "", basename(s), ignore.case = TRUE)
    if (grepl(" - ", base)) return(toupper(trimws(sub(".* - ", "", base))))
    return(NA_character_)
  }
  toupper(s)
}

# Load a team's pitch data from its exported CSV in ~/Downloads. `input` is the
# path produced by resolve_team_path (e.g. ".../NLC2026 - PIT.csv") or any .csv.
load_pitch_data <- function(input) {
  if (!file.exists(input)) {
    stop("load_pitch_data: file not found: ", input,
         "\n  Export the team's tab from its division workbook to ~/Downloads first.")
  }
  readr::read_csv(input, col_types = readr::cols(RTilt = readr::col_character(),
                                                 OTilt = readr::col_character()))
}

# ---- Shared Helper Functions ----

# Compute InZone from PlateX/PlateZ/SzTop/SzBot with ball-radius adjustment
# Matches pipeline/utils.py compute_in_zone (ball-radius rounded-rect,
# validated 0 mismatches vs Statcast zone):
#   dx = max(0, abs(PlateX) - 8.5/12)   distance outside the plate width
#   dz = max(0, SzBot - PlateZ, PlateZ - SzTop)   distance outside the height
#   InZone = "Yes" iff sqrt(dx^2 + dz^2) <= 1.45/12 (one ball radius)
compute_in_zone <- function(plate_x, plate_z, sz_top, sz_bot) {
  ball_radius <- 1.45 / 12  # ~0.121 ft
  half_plate <- 8.5 / 12    # half plate width in feet
  dx <- pmax(0, abs(plate_x) - half_plate)
  dz <- pmax(0, sz_bot - plate_z, plate_z - sz_top)
  case_when(
    is.na(plate_x) | is.na(plate_z) | is.na(sz_top) | is.na(sz_bot) ~ NA_character_,
    sqrt(dx^2 + dz^2) <= ball_radius ~ "Yes",
    TRUE ~ "No"
  )
}

# Helper: sprintf a summary (mean by default) of a numeric vector, or "" when
# it has no non-NA values — an all-NA metric group would otherwise render
# "NaN rpm" / "-Inf mph" / literal NA in the stat tables.
fmt_or_blank <- function(x, fmt, f = mean) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return("")
  sprintf(fmt, f(x))
}

# Helper: average clock-format tilt strings (e.g., "1:54", "12:30")
# Converts H:MM to total minutes on a 12-hour clock, averages circularly, converts back
avg_tilt_clock <- function(tilts) {
  tilts <- na.omit(tilts)
  tilts <- tilts[tilts != ""]
  if (length(tilts) == 0) return(NA_character_)

  # Parse "H:MM" to total minutes (0-720 for 12-hour clock)
  parsed <- sapply(tilts, function(t) {
    parts <- strsplit(as.character(t), ":")[[1]]
    if (length(parts) != 2) return(NA_real_)
    h <- as.numeric(parts[1])
    m <- as.numeric(parts[2])
    if (is.na(h) || is.na(m)) return(NA_real_)
    # Normalize 12 to 0 for circular math
    if (h == 12) h <- 0
    h * 60 + m
  }, USE.NAMES = FALSE)

  parsed <- parsed[!is.na(parsed)]
  if (length(parsed) == 0) return(NA_character_)

  # Circular mean on a 720-minute (12-hour) clock
  angles <- parsed * 2 * pi / 720
  avg_sin <- mean(sin(angles))
  avg_cos <- mean(cos(angles))
  avg_angle <- atan2(avg_sin, avg_cos)
  avg_minutes <- (avg_angle * 720 / (2 * pi)) %% 720

  # Convert back to H:MM
  h <- floor(avg_minutes / 60)
  m <- round(avg_minutes %% 60)
  if (m == 60) { h <- h + 1; m <- 0 }
  if (h == 0) h <- 12
  sprintf("%d:%02d", h, m)
}

# Helper: check if a column has real data for a given pitcher
col_has_data <- function(data, pitcher_name, col_name) {
  if (!(col_name %in% names(data))) return(FALSE)
  pitcher_data <- data %>% filter(Pitcher == pitcher_name)
  any(!is.na(pitcher_data[[col_name]]))
}

# Internal stat column -> the source column it is derived from. A stat column
# whose source is empty for the WHOLE outing carries no information and is
# dropped from the table. Per-pitch gaps are NOT this case: those still render
# through each metric's own blank/"---" handling, because one populated pitch
# anywhere in the outing keeps the column.
#
# Zone%, Chase% and GB% are here because an empty source does not render blank.
# InZone is derived from PlateX/PlateZ/SzTop/SzBot and is all-NA when those are
# missing, so Zone% prints a false "0.0%"; GB% likewise divides real balls in
# play by zero tagged ground balls and prints "0.0%". Both read as measured
# zeros rather than as absent data.
#
# Columns absent from this map are always kept. They derive from Description,
# Pitch Type or the pitch count, none of which can be missing.
STAT_COL_SOURCE <- c(
  avg_velo      = "Velocity",
  max_velo      = "Velocity",
  avg_spin      = "Spin Rate",
  avg_rtilt     = "RTilt",
  avg_tilt      = "OTilt",
  avg_ivb       = "xIndVrtBrk",
  avg_hb        = "xHorzBrk",
  avg_height    = "RelPosZ",
  avg_side      = "RelPosX",
  avg_extension = "Extension",
  avg_arm_angle = "ArmAngle",
  avg_vaa       = "VAA",
  avg_haa       = "HAA",
  iz_percent    = "InZone",
  chase_percent = "InZone",
  gb_percent    = "BBType"
)

# Keep the members of `cols` that still carry data for this pitcher: anything
# with no mapped source, plus every mapped column whose source has at least one
# value somewhere in the outing. Order is preserved.
keep_populated_cols <- function(cols, data, pitcher_name) {
  Filter(function(col) {
    src <- unname(STAT_COL_SOURCE[col])
    if (is.na(src)) return(TRUE)
    col_has_data(data, pitcher_name, src)
  }, cols)
}

# ---- Shared Report Pipeline ----
# Everything below is shared by Daily.R, Season.R, and OnePitcher.R. Report-
# specific behavior is parameterized (header font size, arm-angle lines, plot
# title, per-date pages); genuinely divergent logic (Daily's conditional-column
# stats table, Total row, MLB API statline, platoon splits) stays in the
# individual scripts.

# Read a team CSV and derive InZone from plate location + strike zone bounds.
read_team_pitch_data <- function(input_file) {
  message("Reading pitch data from: ", input_file)
  pitch_data <- load_pitch_data(input_file)

  # Compute InZone from plate location and strike zone boundaries
  # (Description is already simplified by the Python downloader — no re-mapping needed)
  if (all(c("PlateX", "PlateZ", "SzTop", "SzBot") %in% names(pitch_data))) {
    pitch_data$InZone <- compute_in_zone(pitch_data$PlateX, pitch_data$PlateZ,
                                         pitch_data$SzTop, pitch_data$SzBot)
  }
  pitch_data
}

# Apply optional start/end date filters to the pitch data.
# count_messages = TRUE reproduces Daily.R's row-count logging.
filter_by_dates <- function(pitch_data, start_date = NULL, end_date = NULL,
                            count_messages = FALSE) {
  if (is.null(start_date) && is.null(end_date)) return(pitch_data)

  # Make sure Game Date is in date format
  pitch_data$Game_Date_dt <- as.Date(pitch_data$`Game Date`)

  # Filter by start date if provided
  if (!is.null(start_date)) {
    start_date_dt <- as.Date(start_date)
    original_count <- nrow(pitch_data)
    pitch_data <- pitch_data %>% filter(Game_Date_dt >= start_date_dt)
    if (count_messages) {
      message("Start date filtering: ", original_count - nrow(pitch_data),
              " rows removed, ", nrow(pitch_data), " rows remaining")
    }
    message("Filtered data to start from: ", start_date)
  }

  # Filter by end date if provided
  if (!is.null(end_date)) {
    end_date_dt <- as.Date(end_date)
    original_count <- nrow(pitch_data)
    pitch_data <- pitch_data %>% filter(Game_Date_dt <= end_date_dt)
    if (count_messages) {
      message("End date filtering: ", original_count - nrow(pitch_data),
              " rows removed, ", nrow(pitch_data), " rows remaining")
    }
    message("Filtered data to end at: ", end_date)
  }

  pitch_data
}

# Filename suffix for the Season/OnePitcher date-range naming scheme
# (e.g. "_2026-01-01_to_2026-11-28"). Daily.R keeps its own %Y%m%d scheme.
build_date_suffix <- function(start_date = NULL, end_date = NULL) {
  if (is.null(start_date) && is.null(end_date)) return("")
  date_range <- paste0(
    ifelse(is.null(start_date), "Start", start_date),
    "_to_",
    ifelse(is.null(end_date), "End", end_date)
  )
  paste0("_", date_range)
}

# Sanitize "Last, First" into a filename-safe "Last_First".
clean_pitcher_filename <- function(pitcher_name) {
  gsub(", ", "_", pitcher_name) %>%
    gsub(" ", "_", .) %>%
    gsub("[^A-Za-z0-9_]", "", .)
}

# Per-pitch-type summary used by OnePitcher.R (split across its two tables) and
# Season.R (combined table). Formats every value for display. `gb_zero` is the
# GB% shown when there are no (non-bunt) balls in play: "---" (OnePitcher) or
# "0.0%" (Season). Daily.R keeps its own conditional-column variant.
summarize_pitch_type_stats <- function(data, pitcher_name, has_arm_angle = FALSE,
                                       gb_zero = "---") {
  pitcher_data <- data %>%
    filter(Pitcher == pitcher_name)

  total_pitches <- nrow(pitcher_data)
  has_bb_type <- "BBType" %in% names(data)

  # Define pitch outcome event categories (simplified descriptions)
  swing_events <- c("Swinging Strike", "Foul", "In Play")
  csw_events <- c("Called Strike", "Swinging Strike")
  swstr_events <- c("Swinging Strike")
  in_play_events <- c("In Play")

  pitcher_data %>%
    group_by(`Pitch Type`) %>%
    summarize(
      num_thrown = sprintf("%.0f", n()),
      percent_thrown = sprintf("%.1f%%", n() / total_pitches * 100),
      avg_velo = fmt_or_blank(Velocity, "%.1f mph"),
      max_velo = fmt_or_blank(Velocity, "%.1f mph", max),
      avg_spin = fmt_or_blank(`Spin Rate`, "%.0f rpm", function(x) round(mean(x))),
      avg_rtilt = coalesce(avg_tilt_clock(RTilt), ""),
      avg_tilt = coalesce(avg_tilt_clock(OTilt), ""),
      avg_ivb = fmt_or_blank(xIndVrtBrk, "%.1f\""),
      avg_hb = fmt_or_blank(xHorzBrk, "%.1f\""),
      avg_height = fmt_or_blank(RelPosZ, "%.2f'"),
      avg_side = fmt_or_blank(RelPosX, "%.2f'"),
      avg_extension = fmt_or_blank(Extension, "%.2f'"),
      avg_arm_angle = if (has_arm_angle) fmt_or_blank(ArmAngle, "%.1f°") else NA_character_,
      avg_vaa = fmt_or_blank(VAA, "%.2f°"),
      avg_haa = fmt_or_blank(HAA, "%.2f°"),
      # Outcome metrics
      iz_percent = sprintf("%.1f%%", sum(InZone == "Yes", na.rm = TRUE) / n() * 100),
      swing_percent = sprintf("%.1f%%", sum(Description %in% swing_events, na.rm = TRUE) / n() * 100),
      csw_percent = sprintf("%.1f%%", sum(Description %in% csw_events, na.rm = TRUE) / n() * 100),
      swstr_percent = {
        total_swings <- sum(Description %in% swing_events, na.rm = TRUE)
        if (total_swings > 0) {
          sprintf("%.1f%%", sum(Description %in% swstr_events, na.rm = TRUE) / total_swings * 100)
        } else {
          "---"
        }
      },
      # Chase% = swings on pitches outside zone / pitches outside zone
      chase_percent = {
        ooz_pitches <- sum(InZone == "No", na.rm = TRUE)
        if (ooz_pitches > 0) {
          sprintf("%.1f%%", sum(Description %in% swing_events & (InZone == "No"), na.rm = TRUE) / ooz_pitches * 100)
        } else {
          "---"
        }
      },
      gb_percent = {
        if (has_bb_type) {
          total_bip <- sum(Description %in% in_play_events & !grepl("^bunt", BBType), na.rm = TRUE)
          if (total_bip > 0) {
            # Gate on the in-play description too, matching the total_bip
            # denominator — otherwise a ground_ball tagged on a non "In Play"
            # row (e.g. a foul grounder) inflates GB% and can exceed 100%.
            n_gb <- sum(Description %in% in_play_events & BBType == "ground_ball", na.rm = TRUE)
            sprintf("%.1f%%", n_gb / total_bip * 100)
          } else {
            gb_zero
          }
        } else {
          gb_zero
        }
      },
      .groups = "drop"
    )
}

# Map pitch codes to display names, drop unmapped types (never occur in
# practice; an NA name would render blank and sort unpredictably), and sort by
# usage (descending) with exact ties falling back to pitch_order.
map_and_sort_pitch_types <- function(stats_df) {
  stats_df$`Pitch Type` <- pitch_names[stats_df$`Pitch Type`]
  stats_df <- stats_df[!is.na(stats_df$`Pitch Type`), ]
  stats_df$`Pitch Type` <- factor(stats_df$`Pitch Type`, levels = pitch_order)
  stats_df[order(-as.numeric(stats_df$num_thrown), stats_df$`Pitch Type`), ]
}

# Table theme shared by the report stat tables: white cells, bold headers,
# tight horizontal padding so each column is only as wide as its widest content.
make_table_theme <- function(core_fontsize, head_fontsize, pad_h_mm) {
  ttheme_minimal(
    core = list(
      fg_params = list(col = "black", fontsize = core_fontsize),
      bg_params = list(fill = "white"),
      padding = unit(c(2, pad_h_mm), "mm")
    ),
    colhead = list(
      fg_params = list(
        col = "black",
        fontface = "bold",
        fontsize = head_fontsize,
        fontfamily = NULL
      ),
      bg_params = list(fill = "white"),
      padding = unit(c(2, pad_h_mm), "mm")
    )
  )
}

# Format a tableGrob: pitch-type cell colors (Daily's "Total" row gets gray80 /
# bold — a no-op for tables without one), full cell borders, and bold headers
# at `header_fontsize`. `color_platoon_cols = TRUE` (Season.R) also colors the
# duplicate pitch-type column of the 16-column platoon table.
format_table <- function(tbl, stats_df, pitch_names, header_fontsize = 10,
                         color_platoon_cols = FALSE) {
  # Find indices of core-bg cells and core-fg cells (for text)
  bg_indices <- which(tbl$layout$name == "core-bg")
  fg_indices <- which(tbl$layout$name == "core-fg")

  # Color mapping for pitch types
  pitch_colors <- list(
    "FF" = list(fill = alpha("#0086D2"), text = "white"),
    "SL" = list(fill = alpha("#FB6E00"), text = "white"),
    "ST" = list(fill = alpha("#35B6FF"), text = "black"),
    "CU" = list(fill = alpha("#230AA0"), text = "white"),
    "FC" = list(fill = alpha("#A45B16"), text = "white"),
    "SI" = list(fill = alpha("#FFB500"), text = "black"),
    "CH" = list(fill = alpha("#00BA87"), text = "white"),
    "FS" = list(fill = alpha("#F076BA"), text = "black"),
    "KN" = list(fill = alpha("black"), text = "white"),
    "SV" = list(fill = alpha("#A00A55"), text = "white"),
    "EP" = list(fill = alpha("gray50"), text = "white")
  )

  # Color the backgrounds and text for first column cells
  for (i in seq_len(nrow(stats_df))) {
    pitch_full <- stats_df[[1]][i]  # First column (Pitch Type)
    # Find the first matching pitch code
    pitch_code <- names(which(pitch_names == pitch_full))[1]

    # Style the Total row distinctly (Daily.R only; other tables have no Total)
    if (pitch_full == "Total") {
      tbl$grobs[[bg_indices[i]]]$gp$fill <- "gray80"
      tbl$grobs[[fg_indices[i]]]$gp$col <- "black"
      tbl$grobs[[fg_indices[i]]]$gp$font <- 2L  # bold
      next
    }

    # Only proceed if we found a matching pitch code
    if (!is.na(pitch_code) && pitch_code %in% names(pitch_colors)) {
      color_info <- pitch_colors[[pitch_code]]
      tbl$grobs[[bg_indices[i]]]$gp$fill <- color_info$fill
      tbl$grobs[[fg_indices[i]]]$gp$col <- color_info$text
    }
  }

  # For the platoon table, also color the second pitch type column. The two
  # halves shrink whenever keep_populated_cols drops an empty stat column, so
  # locate that column by name instead of assuming the fixed 8+8 layout.
  pitch_type_cols <- which(names(stats_df) == "Pitch Type")
  if (color_platoon_cols && length(pitch_type_cols) >= 2) {
    second_pt <- pitch_type_cols[2]
    for (i in seq_len(nrow(stats_df))) {
      pitch_full <- stats_df[[second_pt]][i]
      pitch_code <- names(which(pitch_names == pitch_full))[1]

      if (!is.na(pitch_code) && pitch_code %in% names(pitch_colors)) {
        color_info <- pitch_colors[[pitch_code]]
        # tableGrob fills core cells column-major, so a column's block starts
        # (column index - 1) * nrow cells in.
        col_offset <- (second_pt - 1) * nrow(stats_df)
        if (length(bg_indices) > (i + col_offset - 1)) {
          tbl$grobs[[bg_indices[i + col_offset]]]$gp$fill <- color_info$fill
          tbl$grobs[[fg_indices[i + col_offset]]]$gp$col <- color_info$text
        }
      }
    }
  }

  # Add borders to all cells
  tbl <- gtable::gtable_add_grob(
    tbl,
    grobs = rectGrob(gp = gpar(fill = NA, col = "black")),
    t = 1, b = nrow(tbl), l = 1, r = ncol(tbl)
  )

  # Add internal borders
  for (i in 1:nrow(tbl)) {
    tbl <- gtable::gtable_add_grob(
      tbl,
      grobs = rectGrob(gp = gpar(fill = NA, col = "black")),
      t = i, b = i, l = 1, r = ncol(tbl)
    )
  }

  for (j in 1:ncol(tbl)) {
    tbl <- gtable::gtable_add_grob(
      tbl,
      grobs = rectGrob(gp = gpar(fill = NA, col = "black")),
      t = 1, b = nrow(tbl), l = j, r = j
    )
  }

  # Force consistent header formatting
  for (i in seq_along(names(stats_df))) {
    header_indices <- which(grepl("colhead", tbl$layout$name))
    if (length(header_indices) >= i) {
      tbl$grobs[[header_indices[i]]]$gp <-
        gpar(col = "black", fontface = "bold", fontsize = header_fontsize)
    }
  }

  return(tbl)
}

# Pitch movement plot shared by all three reports.
#   arm_angle_lines: draw per-pitch-type average arm-angle rays (Season, Daily)
#   show_title:      "First Last Pitch Movement [date]" title (Season,
#                    OnePitcher); Daily uses a separate title grob instead.
create_pitch_plot_shared <- function(pitch_data_filtered, pitcher_name,
                                     game_date = NULL,
                                     arm_angle_lines = FALSE,
                                     show_title = TRUE) {
  title_text <- NULL
  if (show_title) {
    # Format pitcher name for title
    pitcher_name_fmt <- str_replace(pitcher_name, "(.*), (.*)", "\\2 \\1")

    # Create the title with explicit date formatting
    if (is.null(game_date)) {
      title_text <- paste(pitcher_name_fmt, "Pitch Movement")
    } else {
      # Force the game_date to be treated as a string
      title_text <- paste0(pitcher_name_fmt, " Pitch Movement ", as.character(game_date))
    }
  }

  # Create arm angle line data
  arm_angle_segments <- data.frame()

  # Skip the rays entirely when the CSV has no ArmAngle column at all
  # (group_by + mean(ArmAngle) would error, not just produce NAs)
  if (arm_angle_lines && "ArmAngle" %in% names(pitch_data_filtered)) {
    # Calculate average arm angle for each pitch type for arm angle lines
    arm_angle_data <- pitch_data_filtered %>%
      group_by(`Pitch Type`) %>%
      summarise(avg_arm_angle = mean(ArmAngle, na.rm = TRUE), .groups = "drop") %>%
      filter(!is.na(avg_arm_angle))

    # Determine if pitcher is RHP or LHP based on average arm side release
    avg_arm_side <- mean(pitch_data_filtered$RelPosX, na.rm = TRUE)
    # isTRUE guards the all-NA case: mean(NA, na.rm=TRUE) is NaN and `NaN < 0` is
    # NA, which would crash `if (is_rhp)` below (arm angle can be populated while
    # RelPosX awaits backfill). Default to non-RHP when release side is unknown.
    is_rhp <- isTRUE(avg_arm_side < 0)  # RHP have negative RelPosX

    line_length <- 30  # Adjust this value to change line length

    # Only process arm angles if we have valid data
    if (nrow(arm_angle_data) > 0) {
      for (i in 1:nrow(arm_angle_data)) {
        pitch_type_current <- arm_angle_data$`Pitch Type`[i]
        arm_angle_deg <- arm_angle_data$avg_arm_angle[i]

        # Skip if arm_angle_deg is NA (extra safety check)
        if (is.na(arm_angle_deg)) {
          next
        }

        arm_angle_rad <- arm_angle_deg * pi / 180

        # Calculate slope from arm angle
        slope <- tan(arm_angle_rad)

        if (is_rhp) {
          # For RHP: Q1 (positive x, positive y) and Q4 (positive x, negative y)
          x_end <- line_length / sqrt(1 + slope^2)
          y_end <- slope * x_end
        } else {
          # For LHP:
          # If arm angle is positive -> Q2 (negative x, positive y)
          # If arm angle is negative -> Q3 (negative x, negative y)
          if (arm_angle_deg > 0) {
            # Positive angle: line goes to Q2
            x_end <- -line_length / sqrt(1 + slope^2)
            y_end <- abs(slope * x_end)  # Ensure positive y
          } else {
            # Negative angle: line goes to Q3
            x_end <- -line_length / sqrt(1 + slope^2)
            y_end <- -abs(slope * x_end)  # Ensure negative y
          }
        }

        # Add segment data
        arm_angle_segments <- rbind(arm_angle_segments,
                                    data.frame(x = 0, y = 0, xend = x_end, yend = y_end,
                                               `Pitch Type` = pitch_type_current, check.names = FALSE))
      }
    }
  }

  # Create the base plot
  p <- ggplot(
    pitch_data_filtered,
    aes(x = xHorzBrk, y = xIndVrtBrk, color = `Pitch Type`, fill = `Pitch Type`)
  ) +
    geom_hline(yintercept = 0, color = "black", linetype = "dashed", linewidth = 0.5) +
    geom_vline(xintercept = 0, color = "black", linetype = "dashed", linewidth = 0.5)

  # Only add arm angle lines if we have data for them
  if (nrow(arm_angle_segments) > 0) {
    p <- p + geom_segment(data = arm_angle_segments,
                          aes(x = x, y = y, xend = xend, yend = yend, color = `Pitch Type`),
                          linetype = "longdash",
                          alpha = 1,
                          linewidth = 0.8,
                          inherit.aes = FALSE)
  }

  # Add the rest of the plot elements
  p <- p +
    stat_ellipse(geom = "polygon", alpha = 0, level = 0.68, type = "norm", linetype = "longdash") +
    geom_point(size = 3.5, alpha = 1) +
    scale_color_manual(values = pitch_colors) +
    scale_fill_manual(values = pitch_colors) +
    scale_x_continuous(breaks = c(-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)) +
    scale_y_continuous(breaks = c(-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)) +
    { if (show_title) {
        labs(
          title = title_text,
          x = "Horizontal Break (in.)",
          y = "Induced Vertical Break (in.)"
        )
      } else {
        labs(x = "Horizontal Break (in.)", y = "Induced Vertical Break (in.)")
      }
    } +
    coord_cartesian(xlim = c(-25, 25), ylim = c(-25, 25)) +
    theme_minimal(base_size = 5) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
      legend.position = "bottom",
      legend.background = element_rect(fill = "white"),
      plot.title = if (show_title) element_text(face = "bold", hjust = 0.5, size = 24) else element_blank(),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 10),
      plot.margin = margin(t = 20, r = 10, b = 10, l = 10, unit = "pt")
    )

  return(p)
}

# Render the PDF page(s) for one pitcher using the calling script's plot/table
# builders. per_date_pages = TRUE (OnePitcher) gives each game date its own
# page when the pitcher has multiple dates; FALSE (Season) always combines.
render_pitcher_pages <- function(pitcher_data, pitcher_name, per_date_pages,
                                 plot_fun, tables_fun) {
  if (per_date_pages) {
    # Get unique game dates for this pitcher
    game_dates <- unique(pitcher_data$`Game Date`)

    if (length(game_dates) > 1) {
      # If there are multiple dates, create a separate page for each date
      message(paste("Found", length(game_dates), "different game dates for", pitcher_name))

      # Sort game dates chronologically (earliest to latest). as.character
      # matters: `for` over a Date vector strips the class, so log messages
      # would print the numeric day serial ("on 20658") instead of the date.
      game_dates <- as.character(sort(game_dates))

      for (game_date in game_dates) {
        # Filter data for this specific date
        date_data <- pitcher_data %>%
          filter(`Game Date` == game_date)

        # Filter for movement data
        pitch_data_filtered <- date_data %>%
          drop_na(xHorzBrk, xIndVrtBrk)

        if (nrow(pitch_data_filtered) == 0) {
          message(paste("Skipping", pitcher_name, "on", game_date, "- no valid movement data"))
          next
        }

        # Format date as string (YYYY-MM-DD)
        date_string <- format(as.Date(game_date), "%Y-%m-%d")

        # Create the pitch movement plot with date in title
        pitch_plot <- plot_fun(pitch_data_filtered, pitcher_name, date_string)

        # Create the stats tables for this date
        table_plot <- tables_fun(date_data, pitcher_name)

        # Combine plot and table, print to the current PDF page
        print(plot_grid(pitch_plot, table_plot, ncol = 1, rel_heights = c(1.2, 1)))
        message(paste("Created plot for", pitcher_name, "on", game_date))
      }
      return(invisible(TRUE))
    }
  }

  # Single combined page (Season always; OnePitcher when only one game date)
  pitch_data_filtered <- pitcher_data %>%
    drop_na(xHorzBrk, xIndVrtBrk)

  if (nrow(pitch_data_filtered) == 0) {
    message(paste("Skipping", pitcher_name, "- no valid movement data"))
    return(invisible(FALSE))
  }

  # Create the pitch movement plot and stats tables
  pitch_plot <- plot_fun(pitch_data_filtered, pitcher_name)
  table_plot <- tables_fun(pitcher_data, pitcher_name)

  # Combine plot and table, print to the current PDF page
  print(plot_grid(pitch_plot, table_plot, ncol = 1, rel_heights = c(1.2, 1)))
  message(paste("Created plot for", pitcher_name))
  invisible(TRUE)
}

# Shared main driver for Season.R and OnePitcher.R (Daily.R keeps its own:
# team extraction from the filename, MLB API statlines, title grobs, and a
# different filename scheme make it a genuinely different report).
generate_pitcher_reports_core <- function(input_file, output_dir,
                                          pitcher_filter = NULL,
                                          start_date = NULL,
                                          end_date = NULL,
                                          per_date_pages = FALSE,
                                          plot_fun, tables_fun) {
  # Read data; RTilt/OTilt stay character to prevent time parsing — handled
  # inside load_pitch_data().
  pitch_data <- read_team_pitch_data(input_file)

  # Apply date filtering if specified
  pitch_data <- filter_by_dates(pitch_data, start_date, end_date)

  # Generate filename suffix based on date filters
  date_suffix <- build_date_suffix(start_date, end_date)

  # Handle pitcher filtering
  if (!is.null(pitcher_filter)) {
    # If filtering for a specific pitcher, create a single report just for that pitcher
    message("Creating report for specific pitcher: ", pitcher_filter)

    # Filter for a single pitcher
    pitcher_data <- pitch_data %>% filter(Pitcher == pitcher_filter)

    if (nrow(pitcher_data) == 0) {
      message("No data found for pitcher: ", pitcher_filter, " in the specified date range")
      return(NULL)
    }

    # Get the team for the pitcher
    current_team <- unique(pitcher_data$PTeam)[1]

    # Create a cleaner version of the pitcher name for the filename
    clean_pitcher_name <- clean_pitcher_filename(pitcher_filter)

    # Create pitcher-specific PDF with date range if applicable
    pdf_filename <- paste0(output_dir, clean_pitcher_name, date_suffix, "_Pitcher_Report.pdf")
    pdf(pdf_filename, width = 15, height = 18)

    render_pitcher_pages(pitcher_data, pitcher_filter, per_date_pages,
                         plot_fun, tables_fun)

    # Close the PDF
    dev.off()
    message(paste("Pitcher report for", pitcher_filter, "saved to", pdf_filename))

  } else {
    # Original functionality - create team-based reports
    # Get list of all unique teams
    all_teams <- unique(pitch_data$PTeam)
    message("Found ", length(all_teams), " teams in the dataset")

    # Process each team separately
    for (current_team in all_teams) {
      # Create team-specific PDF with date range if applicable
      pdf_filename <- paste0(output_dir, current_team, date_suffix, "_Pitcher_Reports.pdf")
      message("Creating report for team: ", current_team)
      pdf(pdf_filename, width = 15, height = 18)

      # Get pitchers for this team
      all_pitchers <- pitch_data %>%
        filter(PTeam == current_team) %>%
        pull(Pitcher) %>%
        unique() %>%
        sort()

      message("Processing ", length(all_pitchers), " pitchers for ", current_team)

      # Loop through each pitcher. Filter by team AND name (matching
      # Daily.R): a pitcher traded within the division workbook has rows
      # under both PTeams, and name-only filtering would merge the other
      # team's pitches into this team's report.
      for (selected_pitcher in all_pitchers) {
        pitcher_data <- pitch_data %>%
          filter(PTeam == current_team, Pitcher == selected_pitcher)

        render_pitcher_pages(pitcher_data, selected_pitcher, per_date_pages,
                             plot_fun, tables_fun)
      }

      # Close the PDF device for this team
      dev.off()
      message(paste("Pitcher report for team", current_team, "saved to", pdf_filename))
    }

    message("All team pitcher reports have been created")
  }
}
