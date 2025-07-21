# Training Performance Plotting Script
# This script reads training metrics from multiple experiment directories and creates
# comparative plots of policy and value losses across different hyperparameter configurations.

library(ggplot2)
library(dplyr)
library(stringr)
library(purrr)
library(rlang)

# Configuration
root_dir <- "~/Documents/programming/Snowflake2025"
main_res_dir <- file.path(root_dir, "checkpoints/hyperparameter_tuning/")
plots_dir <- file.path(root_dir, "analysis/training_performance/hyperparam_sweep_2025_07_18")
run_tag <- "low_val_lr_2025_07_19"

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

# Function to extract meaningful differences between run names
extract_run_differences <- function(run_names) {
  if (length(run_names) <= 1) {
    return(setNames(run_names, run_names))
  }
  
  # Manual mapping for known runs (add more as needed)
  manual_labels <- c(
    "sweep_run_0_learning_rate0.001_batch_size256_max_grad_norm20_dropout_prob0_weight_decay0.0001_value_learning_rate_factor0.0002_value_weight_decay_factor250.0_20250719_062756" = "wd=0.0001, v_lr_f=0.0002, v_wd_f=250",
    "sweep_run_0_learning_rate0.001_batch_size256_max_grad_norm20_dropout_prob0_weight_decay0.0001_value_learning_rate_factor0.01_value_weight_decay_factor25.0_20250719_052409" = "wd=0.0001, v_lr_f=0.01, v_wd_f=25",
    "sweep_run_0_learning_rate0.001_batch_size256_max_grad_norm20_dropout_prob0_weight_decay0.0002_value_learning_rate_factor0.1_value_weight_decay_factor2.0_20250719_000919" = "wd=0.0002, v_lr_f=0.1, v_wd_f=2",
    "sweep_run_0_learning_rate0.001_batch_size256_max_grad_norm20_dropout_prob0_weight_decay0.0002_value_learning_rate_factor0.1_value_weight_decay_factor2.0_20250719_002329" = "wd=0.0002, v_lr_f=0.1, v_wd_f=2",
    "sweep_run_1_learning_rate0.001_batch_size256_max_grad_norm20_dropout_prob0_weight_decay0.0001_value_learning_rate_factor0.0002_value_weight_decay_factor10.0_20250719_062756" = "wd=0.0001, v_lr_f=0.0002, v_wd_f=10",
    "sweep_run_1_learning_rate0.001_batch_size256_max_grad_norm20_dropout_prob0_weight_decay0.0002_value_learning_rate_factor0.1_value_weight_decay_factor5.0_20250719_002329" = "wd=0.0002, v_lr_f=0.1, v_wd_f=5",
    "sweep_run_2_learning_rate0.001_batch_size256_max_grad_norm20_dropout_prob0_weight_decay0.0002_value_learning_rate_factor0.5_value_weight_decay_factor2.0_20250719_002329" = "wd=0.0002, v_lr_f=0.5, v_wd_f=2",
    "sweep_run_3_learning_rate0.001_batch_size256_max_grad_norm20_dropout_prob0_weight_decay0.0002_value_learning_rate_factor0.5_value_weight_decay_factor5.0_20250719_002329" = "wd=0.0002, v_lr_f=0.5, v_wd_f=5"
  )
  
  # Check if we have manual labels for all runs
  if (all(run_names %in% names(manual_labels))) {
    return(manual_labels[run_names])
  }
  
  # Fallback to automatic extraction for unknown runs
  return(extract_run_differences_auto(run_names))
}

# Automatic extraction function (fallback)
extract_run_differences_auto <- function(run_names) {
  if (length(run_names) <= 1) {
    return(setNames(run_names, run_names))
  }
  
  # Try to extract parameters using regex patterns for structured names
  # Pattern: sweep_run_X_learning_rateY_batch_sizeZ_max_grad_normW_dropout_probA_weight_decayB_value_learning_rate_factorC_value_weight_decay_factorD_timestamp
  unique_labels <- sapply(run_names, function(run_name) {
    # Extract key parameters using regex
    wd_match <- str_extract(run_name, "weight_decay([0-9.]+)")
    v_lr_f_match <- str_extract(run_name, "value_learning_rate_factor([0-9.]+)")
    v_wd_f_match <- str_extract(run_name, "value_weight_decay_factor([0-9.]+)")
    dropout_match <- str_extract(run_name, "dropout_prob([0-9.]+)")
    
    # Build descriptive label
    parts <- character(0)
    
    if (!is.na(wd_match)) {
      wd_val <- str_extract(wd_match, "[0-9.]+")
      parts <- c(parts, paste0("wd=", wd_val))
    }
    
    if (!is.na(v_lr_f_match)) {
      v_lr_val <- str_extract(v_lr_f_match, "[0-9.]+")
      parts <- c(parts, paste0("v_lr_f=", v_lr_val))
    }
    
    if (!is.na(v_wd_f_match)) {
      v_wd_val <- str_extract(v_wd_f_match, "[0-9.]+")
      parts <- c(parts, paste0("v_wd_f=", v_wd_val))
    }
    
    if (!is.na(dropout_match)) {
      dropout_val <- str_extract(dropout_match, "[0-9.]+")
      if (as.numeric(dropout_val) > 0) {
        parts <- c(parts, paste0("dropout=", dropout_val))
      }
    }
    
    if (length(parts) == 0) {
      # Fallback to original logic for unstructured names
      name_parts <- strsplit(run_name, "_")[[1]]
      meaningful_parts <- name_parts[grepl("weight_decay|dropout|value_lr|value_wd|learning_rate|batch_size", name_parts)]
      if (length(meaningful_parts) > 0) {
        return(paste(meaningful_parts, collapse = ", "))
      } else {
        return(paste(name_parts, collapse = "_"))
      }
    }
    
    return(paste(parts, collapse = ", "))
  })
  
  # Ensure uniqueness
  unique_labels <- make.unique(unique_labels, sep = "_")
  
  return(setNames(unique_labels, run_names))
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
    required_cols <- c("epoch", "policy_loss", "val_policy_loss", "value_loss", "val_value_loss")
    missing_cols <- setdiff(required_cols, colnames(data))
    
    if (length(missing_cols) > 0) {
      warning(paste("Missing columns in", file_path, ":", paste(missing_cols, collapse = ", ")))
      return(NULL)
    }
    
    # Ensure epoch is numeric and handle conversion warnings
    data$epoch <- suppressWarnings(as.numeric(data$epoch))
    
    # Also ensure loss columns are numeric
    for (col in c("policy_loss", "val_policy_loss", "value_loss", "val_value_loss")) {
      if (col %in% colnames(data)) {
        data[[col]] <- suppressWarnings(as.numeric(data[[col]]))
      }
    }
    
    # Remove rows with NA values in key columns
    data <- data[complete.cases(data[, required_cols]), ]
    
    if (nrow(data) == 0) {
      warning(paste("No valid data found in", file_path))
      return(NULL)
    }
    
    return(data)
  }, error = function(e) {
    warning(paste("Error reading", file_path, ":", e$message))
    return(NULL)
  })
}

# Function to create loss plot
create_loss_plot <- function(data, loss_type = "policy", title = NULL) {
  if (is.null(title)) {
    title <- paste(str_to_title(loss_type), "Loss by Epoch")
  }
  
  # Determine which columns to use
  train_col <- paste0(loss_type, "_loss")
  val_col <- paste0("val_", loss_type, "_loss")
  
  # Create plot using modern ggplot2 syntax
  plot <- ggplot(data, aes(x = epoch, color = run_label)) +
    geom_line(aes(y = !!sym(train_col)), linewidth = 1) +
    geom_point(aes(y = !!sym(train_col))) +
    geom_line(aes(y = !!sym(val_col)), linetype = "dotted", linewidth = 1) +
    ggtitle(title) +
    labs(
      color = "Configuration",
      x = "Epoch",
      y = paste(str_to_title(loss_type), "Loss")
    ) +
    theme_minimal() +
    theme(
      legend.position = c(0.98, 0.98),
      legend.justification = c("right", "top"),
      legend.background = element_rect(fill = "white", color = "black"),
      legend.box.margin = margin(5, 5, 5, 5)
    )
  
  return(plot)
}

# Function to create bar plot of best validation value losses
create_best_val_value_loss_barplot <- function(combined_data) {
  # Extract the best validation value loss for each run
  best_val_losses <- combined_data %>%
    group_by(run_label, original_dir) %>%
    summarise(
      best_val_value_loss = min(val_value_loss, na.rm = TRUE),
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
      panel.grid.minor = element_blank()
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
    cat(sprintf("   Achieved at Epoch: %d\n", top_3$epoch_of_best[i]))
    cat(sprintf("   Directory: %s\n", top_3$original_dir[i]))
  }
  
  cat("\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  
  return(top_3)
}

# Main execution
main <- function() {
  cat("Starting training performance analysis...\n")
  
  # Find all subdirectories with training metrics
  if (!dir.exists(main_res_dir)) {
    stop(paste("Main results directory not found:", main_res_dir))
  }
  
  all_sub_dirs <- list.dirs(main_res_dir, full.names = FALSE, recursive = FALSE)
  valid_dirs <- character(0)
  
  for (sub_dir in all_sub_dirs) {
    metrics_file <- file.path(main_res_dir, sub_dir, "training_metrics.csv")
    if (file.exists(metrics_file)) {
      valid_dirs <- c(valid_dirs, sub_dir)
    }
  }
  
  if (length(valid_dirs) == 0) {
    stop("No training_metrics.csv files found in any subdirectories")
  }
  
  cat("Found", length(valid_dirs), "directories with training metrics\n")
  
  # Extract meaningful run labels
  run_labels <- extract_run_differences(valid_dirs)
  
  # Read all training data
  all_data <- list()
  successful_reads <- 0
  
  for (sub_dir in valid_dirs) {
    metrics_file <- file.path(main_res_dir, sub_dir, "training_metrics.csv")
    data <- read_training_metrics(metrics_file)
    
    if (!is.null(data)) {
      data$run_label <- run_labels[sub_dir]
      data$original_dir <- sub_dir
      all_data[[sub_dir]] <- data
      successful_reads <- successful_reads + 1
      cat("Successfully read:", sub_dir, "->", run_labels[sub_dir], "\n")
    }
  }
  
  if (successful_reads == 0) {
    stop("No valid training data could be read")
  }
  
  cat("Successfully read data from", successful_reads, "runs\n")
  
  # Combine all data
  combined_data <- bind_rows(all_data)
  
  # Create plots
  cat("Creating plots...\n")
  
  # Policy loss plot
  policy_plot <- create_loss_plot(combined_data, "policy", "Policy Loss by Epoch")
  policy_filename <- get_unique_filename(file.path(plots_dir, paste0("policy_loss_by_epoch_", run_tag)))
  ggsave(policy_filename, plot = policy_plot, width = 10, height = 6, dpi = 150)
  cat("Saved policy loss plot:", policy_filename, "\n")
  
  # Value loss plot
  value_plot <- create_loss_plot(combined_data, "value", "Value Loss by Epoch")
  value_filename <- get_unique_filename(file.path(plots_dir, paste0("value_loss_by_epoch_", run_tag)))
  ggsave(value_filename, plot = value_plot, width = 10, height = 6, dpi = 150)
  cat("Saved value loss plot:", value_filename, "\n")
  
  # Create best validation value loss bar plot
  cat("Creating best validation value loss bar plot...\n")
  best_val_plot_result <- create_best_val_value_loss_barplot(combined_data)
  best_val_plot <- best_val_plot_result$plot
  best_val_losses <- best_val_plot_result$data
  
  best_val_filename <- get_unique_filename(file.path(plots_dir, paste0("best_val_value_loss_barplot_", run_tag)))
  ggsave(best_val_filename, plot = best_val_plot, width = 12, height = 8, dpi = 150)
  cat("Saved best validation value loss bar plot:", best_val_filename, "\n")
  
  # Print top 3 best models
  top_3_models <- print_top_3_models(best_val_losses)
  
  # Print summary
  cat("\nSummary:\n")
  cat("- Total runs found:", length(valid_dirs), "\n")
  cat("- Successful reads:", successful_reads, "\n")
  cat("- Unique configurations:", length(unique(combined_data$run_label)), "\n")
  cat("- Total epochs across all runs:", nrow(combined_data), "\n")
  cat("- Best overall validation value loss:", min(best_val_losses$best_val_value_loss), "\n")
  cat("- Worst overall validation value loss:", max(best_val_losses$best_val_value_loss), "\n")
  
  # Return the combined data and best losses for further analysis if needed
  invisible(list(
    combined_data = combined_data,
    best_val_losses = best_val_losses,
    top_3_models = top_3_models
  ))
}

# Run the main function
result <- main()
