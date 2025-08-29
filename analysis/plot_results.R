# Training Performance Plotting Script
# This script reads training metrics from the centralized bookkeeping directory and creates
# comparative plots of policy and value losses across different hyperparameter configurations.

library(ggplot2)
library(dplyr)
library(stringr)
library(purrr)
library(rlang)

# Configuration
root_dir <- "~/Documents/programming/Snowflake2025"
bookkeeping_dir <- file.path(root_dir, "checkpoints/bookkeeping/")
plots_dir <- file.path(root_dir, "analysis/training_performance/hyperparam_sweep_2025_08_28")
run_tag <- "aug28th_extraValueLayer_sweep_2025_08_28"

# Look for all training metrics files from the recent sweep
METRICS_FILES <- list.files(bookkeeping_dir, pattern = "training_metrics_20250828_18\\.csv$", full.names = TRUE)
# 


# Ensure plots directory exists
if (!dir.exists(plots_dir)) {
  dir.create(plots_dir, recursive = TRUE)
}

# Function to find unique plot filename
get_unique_filename <- function(base_path, extension = ".png") {
  if (!file.exists(paste0(base_path, extension))) {
    return(paste0(base_path, extension))
  }
  
  counter <- 1
  while (file.exists(paste0(base_path, "_", counter, extension))) {
    counter <- counter + 1
  }
  return(paste0(base_path, "_", counter, extension))
}

# Function to extract meaningful differences between experiment configurations
extract_experiment_differences <- function(data) {
  # Get unique combinations of the varying parameters
  varying_params <- c("learning_rate", "max_grad_norm", "value_learning_rate_factor")
  
  # Create a unique identifier for each configuration
  config_groups <- data %>%
    select(all_of(varying_params)) %>%
    distinct() %>%
    mutate(
      config_id = row_number(),
      # Create readable labels
      label = paste0(
        "lr=", sprintf("%.5f", learning_rate),
        ", mgn=", max_grad_norm,
        ", vlrf=", value_learning_rate_factor
      )
    )
  
  # Join back to original data
  result <- data %>%
    left_join(config_groups, by = varying_params) %>%
    mutate(run_label = label)
  
  return(result)
}

# Function to read and validate training metrics
read_training_metrics <- function(file_path) {
  if (!file.exists(file_path)) {
    warning(paste("File not found:", file_path))
    return(NULL)
  }
  
  tryCatch({
    data <- read.csv(file_path, stringsAsFactors = FALSE)
    
    # Check for required columns
    required_cols <- c("epoch", "policy_loss", "value_loss")
    missing_cols <- setdiff(required_cols, colnames(data))
    
    if (length(missing_cols) > 0) {
      warning(paste("Missing columns in", file_path, ":", paste(missing_cols, collapse = ", ")))
      return(NULL)
    }
    
    # Handle epoch parsing - extract both epoch and mini-epoch from strings like "3_mini4"
    data$epoch_numeric <- sapply(data$epoch, function(x) {
      if (is.na(x) || x == "") return(NA)
      # Extract the main epoch number (before the underscore)
      parts <- strsplit(as.character(x), "_")[[1]]
      if (length(parts) > 0) {
        # Try to extract numeric part
        num_part <- as.numeric(parts[1])
        if (!is.na(num_part)) return(num_part)
      }
      return(NA)
    })
    
    # Extract mini-epoch number from strings like "3_mini4"
    data$mini_epoch_numeric <- sapply(data$epoch, function(x) {
      if (is.na(x) || x == "") return(NA)
      # Look for "mini" followed by a number
      mini_match <- str_extract(as.character(x), "mini([0-9]+)")
      if (!is.na(mini_match)) {
        # Extract the number after "mini"
        mini_num <- as.numeric(str_extract(mini_match, "[0-9]+"))
        if (!is.na(mini_num)) return(mini_num)
      }
      return(NA)
    })
    
    # Also ensure loss columns are numeric
    for (col in c("policy_loss", "val_policy_loss", "value_loss", "val_value_loss")) {
      if (col %in% colnames(data)) {
        data[[col]] <- suppressWarnings(as.numeric(data[[col]]))
      }
    }
    
    # Remove rows with NA values in key columns
    data <- data[complete.cases(data[, c("epoch_numeric", "mini_epoch_numeric", "policy_loss", "value_loss")]), ]
    
    if (nrow(data) == 0) {
      warning(paste("No valid data found in", file_path))
      return(NULL)
    }
    
    # Replace epoch with numeric version and add mini_epoch
    data$epoch <- data$epoch_numeric
    data$mini_epoch <- data$mini_epoch_numeric
    data$epoch_numeric <- NULL
    data$mini_epoch_numeric <- NULL
    
    return(data)
  }, error = function(e) {
    warning(paste("Error reading", file_path, ":", e$message))
    return(NULL)
  })
}

# Helper function to handle x-axis selection for plots
get_plot_x_axis <- function(data) {
  # Check if we have multiple epochs with the same mini_epoch values
  epoch_mini_epoch_counts <- data %>%
    group_by(run_label, mini_epoch) %>%
    summarise(n_epochs = n_distinct(epoch), .groups = 'drop') %>%
    filter(n_epochs > 1)
  
  if (nrow(epoch_mini_epoch_counts) > 0) {
    warning("Multiple epochs found for same mini_epoch values. Creating composite x-axis.")
    
    # Create a composite x-axis that includes both epoch and mini_epoch
    plot_data <- data %>%
      mutate(
        # Create a composite x value: epoch + mini_epoch/100
        # This ensures unique x values while preserving the relationship
        composite_x = epoch + mini_epoch/100
      )
    
    return(list(
      data = plot_data,
      x_var = "composite_x",
      x_label = "Epoch.Mini-Epoch"
    ))
  } else {
    return(list(
      data = data,
      x_var = "mini_epoch",
      x_label = "Mini-Epoch"
    ))
  }
}

# Function to create loss plot
create_loss_plot <- function(data, loss_type = "policy", title = NULL) {
  if (is.null(title)) {
    title <- paste(str_to_title(loss_type), "Loss by Mini-Epoch")
  }
  
  # Determine which columns to use
  train_col <- paste0(loss_type, "_loss")
  val_col <- paste0("val_", loss_type, "_loss")
  
  # Get appropriate x-axis configuration
  x_config <- get_plot_x_axis(data)
  plot_data <- x_config$data
  
  # Create plot using the appropriate x-axis
  plot <- ggplot(plot_data, aes(x = !!sym(x_config$x_var), color = run_label)) +
    geom_line(aes(y = !!sym(train_col)), linewidth = 1) +
    geom_point(aes(y = !!sym(train_col))) +
    geom_line(aes(y = !!sym(val_col)), linetype = "dotted", linewidth = 1) +
    ggtitle(title) +
    labs(
      color = "Configuration",
      x = x_config$x_label,
      y = paste(str_to_title(loss_type), "Loss")
    ) +
    theme_minimal() +
    theme(
      legend.position = c(0.98, 0.98),
      legend.justification = c("right", "top"),
      legend.background = element_rect(fill = "white", color = "black"),
      legend.box.margin = margin(5, 5, 5, 5),
      panel.background = element_rect(fill = "white"),
      plot.background = element_rect(fill = "white")
    )
  
  return(plot)
}

# Function to create gradient clipping plot
create_gradient_clipping_plot <- function(data) {
  # Check if gradient columns exist
  if (!all(c("gradient_norm", "post_clip_gradient_norm", "max_grad_norm") %in% colnames(data))) {
    warning("Gradient clipping columns not found in data")
    return(NULL)
  }
  
  # Convert to numeric and filter out NA values
  plot_data <- data %>%
    mutate(
      gradient_norm = as.numeric(gradient_norm),
      post_clip_gradient_norm = as.numeric(post_clip_gradient_norm),
      max_grad_norm = as.numeric(max_grad_norm)
    ) %>%
    filter(!is.na(gradient_norm) & !is.na(post_clip_gradient_norm))
  
  if (nrow(plot_data) == 0) {
    warning("No valid gradient clipping data found")
    return(NULL)
  }
  
  # Get appropriate x-axis configuration
  x_config <- get_plot_x_axis(plot_data)
  plot_data <- x_config$data
  
  # Create plot using the appropriate x-axis
  plot <- ggplot(plot_data, aes(x = !!sym(x_config$x_var), color = run_label)) +
    geom_line(aes(y = gradient_norm), linewidth = 1, linetype = "solid") +
    geom_line(aes(y = post_clip_gradient_norm), linewidth = 1, linetype = "dashed") +
    geom_hline(aes(yintercept = max_grad_norm), linetype = "dotted", color = "red", linewidth = 1) +
    ggtitle("Gradient Norms: Pre-clip vs Post-clip") +
    labs(
      color = "Configuration",
      x = x_config$x_label,
      y = "Gradient Norm",
      caption = "Solid = Pre-clip, Dashed = Post-clip, Red dotted = Max gradient norm"
    ) +
    theme_minimal() +
    theme(
      legend.position = c(0.98, 0.98),
      legend.justification = c("right", "top"),
      legend.background = element_rect(fill = "white", color = "black"),
      legend.box.margin = margin(5, 5, 5, 5),
      panel.background = element_rect(fill = "white"),
      plot.background = element_rect(fill = "white")
    )
  
  return(plot)
}

# Function to create learning rate statistics plot
create_lr_stats_plot <- function(data) {
  # Check if LR columns exist
  if (!all(c("lr_mean", "lr_min", "lr_max") %in% colnames(data))) {
    warning("Learning rate statistics columns not found in data")
    return(NULL)
  }
  
  # Convert to numeric and filter out NA values
  plot_data <- data %>%
    mutate(
      lr_mean = as.numeric(lr_mean),
      lr_min = as.numeric(lr_min),
      lr_max = as.numeric(lr_max)
    ) %>%
    filter(!is.na(lr_mean))
  
  if (nrow(plot_data) == 0) {
    warning("No valid learning rate data found")
    return(NULL)
  }
  
  # Get appropriate x-axis configuration
  x_config <- get_plot_x_axis(plot_data)
  plot_data <- x_config$data
  
  # Create plot using the appropriate x-axis
  plot <- ggplot(plot_data, aes(x = !!sym(x_config$x_var), color = run_label)) +
    geom_line(aes(y = lr_mean), linewidth = 1) +
    geom_ribbon(aes(ymin = lr_min, ymax = lr_max, fill = run_label), alpha = 0.2) +
    ggtitle("Learning Rate Statistics by Mini-Epoch") +
    labs(
      color = "Configuration",
      fill = "Configuration",
      x = x_config$x_label,
      y = "Learning Rate",
      caption = "Line = Mean, Shaded area = Min to Max range"
    ) +
    theme_minimal() +
    theme(
      legend.position = c(0.98, 0.98),
      legend.justification = c("right", "top"),
      legend.background = element_rect(fill = "white", color = "black"),
      legend.box.margin = margin(5, 5, 5, 5),
      panel.background = element_rect(fill = "white"),
      plot.background = element_rect(fill = "white")
    )
  
  return(plot)
}

# Function to create weight statistics plot
create_weight_stats_plot <- function(data) {
  # Check if weight columns exist
  if (!all(c("weight_norm_mean", "weight_norm_std") %in% colnames(data))) {
    warning("Weight statistics columns not found in data")
    return(NULL)
  }
  
  # Convert to numeric and filter out NA values
  plot_data <- data %>%
    mutate(
      weight_norm_mean = as.numeric(weight_norm_mean),
      weight_norm_std = as.numeric(weight_norm_std)
    ) %>%
    filter(!is.na(weight_norm_mean))
  
  if (nrow(plot_data) == 0) {
    warning("No valid weight statistics data found")
    return(NULL)
  }
  
  # Get appropriate x-axis configuration
  x_config <- get_plot_x_axis(plot_data)
  plot_data <- x_config$data
  
  # Create plot using the appropriate x-axis
  plot <- ggplot(plot_data, aes(x = !!sym(x_config$x_var), color = run_label)) +
    geom_line(aes(y = weight_norm_mean), linewidth = 1) +
    geom_ribbon(aes(ymin = weight_norm_mean - weight_norm_std, 
                    ymax = weight_norm_mean + weight_norm_std, 
                    fill = run_label), alpha = 0.2) +
    ggtitle("Weight Norm Statistics by Mini-Epoch") +
    labs(
      color = "Configuration",
      fill = "Configuration",
      x = x_config$x_label,
      y = "Weight Norm",
      caption = "Line = Mean, Shaded area = ±1 standard deviation"
    ) +
    theme_minimal() +
    theme(
      legend.position = c(0.98, 0.98),
      legend.justification = c("right", "top"),
      legend.background = element_rect(fill = "white", color = "black"),
      legend.box.margin = margin(5, 5, 5, 5),
      panel.background = element_rect(fill = "white"),
      plot.background = element_rect(fill = "white")
    )
  
  return(plot)
}

# Function to create gradient statistics plot
create_gradient_stats_plot <- function(data) {
  # Check if gradient statistics columns exist
  if (!all(c("gradient_norm_mean", "gradient_norm_std") %in% colnames(data))) {
    warning("Gradient statistics columns not found in data")
    return(NULL)
  }
  
  # Convert to numeric and filter out NA values
  plot_data <- data %>%
    mutate(
      gradient_norm_mean = as.numeric(gradient_norm_mean),
      gradient_norm_std = as.numeric(gradient_norm_std)
    ) %>%
    filter(!is.na(gradient_norm_mean))
  
  if (nrow(plot_data) == 0) {
    warning("No valid gradient statistics data found")
    return(NULL)
  }
  
  # Get appropriate x-axis configuration
  x_config <- get_plot_x_axis(plot_data)
  plot_data <- x_config$data
  
  # Create plot using the appropriate x-axis
  plot <- ggplot(plot_data, aes(x = !!sym(x_config$x_var), color = run_label)) +
    geom_line(aes(y = gradient_norm_mean), linewidth = 1) +
    geom_ribbon(aes(ymin = gradient_norm_mean - gradient_norm_std, 
                    ymax = gradient_norm_mean + gradient_norm_std, 
                    fill = run_label), alpha = 0.2) +
    ggtitle("Gradient Norm Statistics by Mini-Epoch") +
    labs(
      color = "Configuration",
      fill = "Configuration",
      x = x_config$x_label,
      y = "Gradient Norm",
      caption = "Line = Mean, Shaded area = ±1 standard deviation"
    ) +
    theme_minimal() +
    theme(
      legend.position = c(0.98, 0.98),
      legend.justification = c("right", "top"),
      legend.background = element_rect(fill = "white", color = "black"),
      legend.box.margin = margin(5, 5, 5, 5),
      panel.background = element_rect(fill = "white"),
      plot.background = element_rect(fill = "white")
    )
  
  return(plot)
}

# Function to create bar plot of best validation value losses
create_best_val_value_loss_barplot <- function(combined_data) {
  # Extract the best validation value loss for each run
  best_val_losses <- combined_data %>%
    group_by(run_label) %>%
    summarise(
      best_val_value_loss = min(val_value_loss, na.rm = TRUE),
      mini_epoch_of_best = mini_epoch[which.min(val_value_loss)],
      epoch_of_best = epoch[which.min(val_value_loss)],
      .groups = 'drop'
    ) %>%
    arrange(best_val_value_loss)
  
  # Create bar plot
  plot <- ggplot(best_val_losses, aes(x = reorder(run_label, best_val_value_loss), y = best_val_value_loss)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
    geom_text(aes(label = sprintf("%.4f", best_val_value_loss)), 
              position = position_stack(vjust = 0.5), 
              size = 3, color = "white", fontface = "bold") +
    coord_flip() +  # Horizontal bars for better label readability
    ggtitle("Best Validation Value Loss by Model Configuration") +
    labs(
      x = "Model Configuration",
      y = "Best Validation Value Loss"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 8),
      plot.title = element_text(size = 14, face = "bold"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white"),
      plot.background = element_rect(fill = "white")
    )
  
  return(list(plot = plot, data = best_val_losses))
}

# Function to print top 3 best models
print_top_3_models <- function(best_val_losses) {
  cat("\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat("TOP 3 BEST MODELS BY VALIDATION VALUE LOSS\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  
  top_3 <- head(best_val_losses, 3)
  
  for (i in 1:nrow(top_3)) {
    cat(sprintf("\n%d. %s\n", i, top_3$run_label[i]))
    cat(sprintf("   Best Validation Value Loss: %.6f\n", top_3$best_val_value_loss[i]))
    cat(sprintf("   Achieved at Epoch %d, Mini-Epoch %d\n", top_3$epoch_of_best[i], top_3$mini_epoch_of_best[i]))
  }
  
  cat("\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  
  return(top_3)
}

# Main execution
main <- function() {
  cat("Starting training performance analysis...\n")
  
  # Find the most recent training metrics file
  if (!dir.exists(bookkeeping_dir)) {
    stop(paste("Bookkeeping directory not found:", bookkeeping_dir))
  }
  
  if (length(METRICS_FILES) == 0) {
    stop("No training_metrics CSV files found in bookkeeping directory")
  }
  
  # Sort by modification time
  file_info <- file.info(METRICS_FILES)
  METRICS_FILES <- METRICS_FILES[order(file_info$mtime, decreasing = TRUE)]
  
  cat("Found", length(METRICS_FILES), "training metrics files\n")
  
  # Read all training data
  all_data <- list()
  successful_reads <- 0
  
  for (file_path in METRICS_FILES) {
    cat("Reading:", basename(file_path), "\n")
    data <- read_training_metrics(file_path)
    
    if (!is.null(data)) {
      data$source_file <- basename(file_path)
      all_data[[basename(file_path)]] <- data
      successful_reads <- successful_reads + 1
      cat("  Successfully read", nrow(data), "rows\n")
    }
  }
  
  if (successful_reads == 0) {
    stop("No valid training data could be read")
  }
  
  # Combine all data
  combined_data <- bind_rows(all_data)
  cat("Combined data has", nrow(combined_data), "total rows\n")
  
  # Extract experiment differences and create run labels
  data_with_labels <- extract_experiment_differences(combined_data)
  
  # Filter to only include rows with validation data (mini-epoch rows)
  validation_data <- data_with_labels %>%
    filter(!is.na(val_value_loss) & val_value_loss > 0)
  
  if (nrow(validation_data) == 0) {
    stop("No validation data found in the metrics files")
  }
  
  cat("Found", nrow(validation_data), "validation data points\n")
  cat("Unique configurations:", length(unique(validation_data$run_label)), "\n")
  
  # Create plots
  cat("Creating plots...\n")
  
  # Policy loss plot
  policy_plot <- create_loss_plot(validation_data, "policy", "Policy Loss by Mini-Epoch")
  policy_filename <- get_unique_filename(file.path(plots_dir, paste0("policy_loss_by_mini_epoch_", run_tag)))
  ggsave(policy_filename, plot = policy_plot, width = 10, height = 6, dpi = 150)
  cat("Saved policy loss plot:", policy_filename, "\n")
  
  # Value loss plot
  value_plot <- create_loss_plot(validation_data, "value", "Value Loss by Mini-Epoch")
  value_filename <- get_unique_filename(file.path(plots_dir, paste0("value_loss_by_mini_epoch_", run_tag)))
  ggsave(value_filename, plot = value_plot, width = 10, height = 6, dpi = 150)
  cat("Saved value loss plot:", value_filename, "\n")
  
  # Create best validation value loss bar plot
  cat("Creating best validation value loss bar plot...\n")
  best_val_plot_result <- create_best_val_value_loss_barplot(validation_data)
  best_val_plot <- best_val_plot_result$plot
  best_val_losses <- best_val_plot_result$data
  
  best_val_filename <- get_unique_filename(file.path(plots_dir, paste0("best_val_value_loss_barplot_", run_tag)))
  ggsave(best_val_filename, plot = best_val_plot, width = 12, height = 8, dpi = 150)
  cat("Saved best validation value loss bar plot:", best_val_filename, "\n")
  
  # Create diagnostic plots
  cat("Creating diagnostic plots...\n")
  
  # Gradient clipping plot
  gradient_clipping_plot <- create_gradient_clipping_plot(validation_data)
  if (!is.null(gradient_clipping_plot)) {
    gradient_clipping_filename <- get_unique_filename(file.path(plots_dir, paste0("gradient_clipping_", run_tag)))
    ggsave(gradient_clipping_filename, plot = gradient_clipping_plot, width = 10, height = 6, dpi = 150)
    cat("Saved gradient clipping plot:", gradient_clipping_filename, "\n")
  }
  
  # Learning rate statistics plot
  lr_stats_plot <- create_lr_stats_plot(validation_data)
  if (!is.null(lr_stats_plot)) {
    lr_stats_filename <- get_unique_filename(file.path(plots_dir, paste0("learning_rate_stats_", run_tag)))
    ggsave(lr_stats_filename, plot = lr_stats_plot, width = 10, height = 6, dpi = 150)
    cat("Saved learning rate statistics plot:", lr_stats_filename, "\n")
  }
  
  # Weight statistics plot
  weight_stats_plot <- create_weight_stats_plot(validation_data)
  if (!is.null(weight_stats_plot)) {
    weight_stats_filename <- get_unique_filename(file.path(plots_dir, paste0("weight_stats_", run_tag)))
    ggsave(weight_stats_filename, plot = weight_stats_plot, width = 10, height = 6, dpi = 150)
    cat("Saved weight statistics plot:", weight_stats_filename, "\n")
  }
  
  # Gradient statistics plot
  gradient_stats_plot <- create_gradient_stats_plot(validation_data)
  if (!is.null(gradient_stats_plot)) {
    gradient_stats_filename <- get_unique_filename(file.path(plots_dir, paste0("gradient_stats_", run_tag)))
    ggsave(gradient_stats_filename, plot = gradient_stats_plot, width = 10, height = 6, dpi = 150)
    cat("Saved gradient statistics plot:", gradient_stats_filename, "\n")
  }
  
  # Print top 3 best models
  top_3_models <- print_top_3_models(best_val_losses)
  
  # Print summary
  cat("\nSummary:\n")
  cat("- Total validation data points:", nrow(validation_data), "\n")
  cat("- Unique configurations:", length(unique(validation_data$run_label)), "\n")
  cat("- Best overall validation value loss:", min(best_val_losses$best_val_value_loss), "\n")
  cat("- Worst overall validation value loss:", max(best_val_losses$best_val_value_loss), "\n")
  
  # Show configuration details
  cat("\nConfiguration Details:\n")
  config_summary <- validation_data %>%
    select(learning_rate, max_grad_norm, value_learning_rate_factor, run_label) %>%
    distinct() %>%
    arrange(run_label)
  
  for (i in 1:nrow(config_summary)) {
    cat(sprintf("%d. %s\n", i, config_summary$run_label[i]))
  }
  
  # Show data sources
  cat("\nData Sources:\n")
  source_summary <- validation_data %>%
    group_by(source_file) %>%
    summarise(
      n_rows = n(),
      n_configs = n_distinct(run_label),
      .groups = 'drop'
    ) %>%
    arrange(desc(n_rows))
  
  for (i in 1:nrow(source_summary)) {
    cat(sprintf("%d. %s: %d rows, %d configs\n", 
                i, source_summary$source_file[i], 
                source_summary$n_rows[i], 
                source_summary$n_configs[i]))
  }
  
  # Return the data for further analysis if needed
  invisible(list(
    validation_data = validation_data,
    best_val_losses = best_val_losses,
    top_3_models = top_3_models,
    config_summary = config_summary,
    source_summary = source_summary
  ))
}

# Run the main function
result <- main()
