# Mediation analysis 
# ==============================================================================

# Part 1: Ab as exposure (X), TA as mediator (M), and Group as outcome (Y).

# Use logistic outcome regression; extract a-path t-test and b-, c-, c-prime-path z-test p-values.

# ==============================================================================

rm(list=ls())
library(mediation)
library(tidyverse)
library(doRNG)
library(doParallel)
set.seed(2026)

# 1. Load required packages.

packages <- c("R.matlab", "mediation", "foreach", "doParallel")
for (p in packages) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}

# 2. Load  data.

mat_data <- readMat("/Users/linqing/Desktop/Projects/2024 structure_FC/RMN network/Representation mapping network/6.Disease/DK68/mediation/AD_data.mat") # replace your path

dat_harmonized <- mat_data$dat.harmonized # Dimensions: n_regions x n_terms x n_subjects.
Ab_data <- mat_data$Ab.data               # Dimensions: n_subjects x n_regions.
age <- as.numeric(mat_data$age)           # Length: n_subjects.
sex <- as.numeric(mat_data$sex)           # Length: n_subjects.
disease <- as.numeric(mat_data$disease)   # Length: n_subjects; outcome Y.

# Ensure Group is coded as 0/1 for binomial regression.

# For 1/2 coding, map the smaller value to 0 and the larger value to 1.

if (!all(disease %in% c(0, 1))) {
  disease <- ifelse(
    disease == min(disease, na.rm = TRUE),
    0,
    1
  )
  cat("Note: disease has been recoded as 0/1 for logistic regression.\n")
}

# Load region and term labels.

region_info <- read.csv(
  "/Users/linqing/Desktop/Projects/2024 structure_FC/RMN network/Representation mapping network/6.Disease/DK68/DK68_name.csv",
  stringsAsFactors = FALSE
)

term_info <- read.csv(
  "/Users/linqing/Desktop/Projects/2025_RMN/data/Term.csv",
  stringsAsFactors = FALSE
)

region_names <- region_info$label
term_names <- term_info$Term

# 3. Configure the parallel backend.

num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

cat("Parallel processing enabled. Number of cores:", num_cores, "\n")

# ==============================================================================

# 4. Load the selected region-term pairs.

# ==============================================================================

pair_index_file <- "/Users/linqing/Desktop/Projects/2024 structure_FC/RMN network/Representation mapping network/6.Disease/DK68/mediation/AD_mediation_index.csv"

pair_df <- read.csv(
  pair_index_file,
  header = TRUE,
  stringsAsFactors = FALSE
)

# Use the first column as the region index and the second as the term index (R uses 1-based indices).

task_grid <- data.frame(
  region_index = as.numeric(pair_df[, 1]),
  term_index   = as.numeric(pair_df[, 2])
)

total_tasks <- nrow(task_grid)

cat("Number of mediation models to run:", total_tasks, "\n")

# 5. Run models in parallel.

results_df <- foreach(
  k = 1:total_tasks,
  .combine = rbind,
  .packages = c("mediation")
) %dorng% {
  
  i <- task_grid$region_index[k]
  j <- task_grid$term_index[k]
  
  TA_val <- dat_harmonized[i, j, ]
  Ab_val <- Ab_data[, i]
  
  # Build the analysis data.
  # X = Ab
  # M = TA
  # Y = Group
  
  df_temp <- data.frame(
    TA = TA_val,     # M: mediator.
    Ab = Ab_val,     # X: exposure.
    Group = disease, # Y: outcome.
    Age = age,
    Sex = sex
  )
  
  res_row <- data.frame(
    region_index = i,
    region = region_names[i],
    term_index = j,
    term = term_names[j],
    
    # Path coefficients a, b, c, and c-prime and their p-values.
    a_path_est = NA_real_,
    a_path_p = NA_real_,
    b_path_est = NA_real_,
    b_path_p = NA_real_,
    c_prime_path_est = NA_real_,
    c_prime_path_p = NA_real_,
    c_path_est = NA_real_,
    c_path_p = NA_real_,
    
    # Bootstrap mediation estimates.
    ACME_Estimate = NA_real_,
    ACME_CI_Lower = NA_real_,
    ACME_CI_Upper = NA_real_,
    ACME_p_value = NA_real_,
    
    ADE_Estimate = NA_real_,
    ADE_CI_Lower = NA_real_,
    ADE_CI_Upper = NA_real_,
    ADE_p_value = NA_real_,
    
    Total_Estimate = NA_real_,
    Total_CI_Lower = NA_real_,
    Total_CI_Upper = NA_real_,
    Total_p_value = NA_real_,
    
    Prop_Estimate = NA_real_,
    Prop_CI_Lower = NA_real_,
    Prop_CI_Upper = NA_real_,
    Prop_p_value = NA_real_,
    
    Error_Message = "",
    stringsAsFactors = FALSE
  )
  
  # Capture model errors using try.
  
  try_out <- try({
    
    # 1. Mediator regression (a path).
    # Ab -> TA
    # TA is continuous; fit a linear regression.
    
    med.fit <- lm(
      TA ~ Ab + Age + Sex,
      data = df_temp
    )
    
    # 2. Outcome regression (b and c-prime paths).
    # TA -> Group | Ab
    # Ab -> Group | TA
    # Group is binary; fit a logistic regression.
    
    out.fit <- glm(
      Group ~ TA + Ab + Age + Sex,
      data = df_temp,
      family = binomial(link = "logit")
    )
    
    # 3. Total-effect regression (c path).
    # Ab -> Group
    # Group is binary; fit a logistic regression.
    
    total.fit <- glm(
      Group ~ Ab + Age + Sex,
      data = df_temp,
      family = binomial(link = "logit")
    )
    
    # Extract path coefficients a, b, c-prime, and c.
    
    sum_med <- summary(med.fit)$coefficients
    sum_out <- summary(out.fit)$coefficients
    sum_total <- summary(total.fit)$coefficients
    
    # a path: Ab -> TA.
    res_row$a_path_est <- sum_med["Ab", "Estimate"]
    res_row$a_path_p   <- sum_med["Ab", "Pr(>|t|)"]
    
    # b path: TA -> Group, controlling for Ab, Age, and Sex.
    res_row$b_path_est <- sum_out["TA", "Estimate"]
    res_row$b_path_p   <- sum_out["TA", "Pr(>|z|)"]
    
    # c-prime path: Ab -> Group, controlling for TA, Age, and Sex.
    res_row$c_prime_path_est <- sum_out["Ab", "Estimate"]
    res_row$c_prime_path_p   <- sum_out["Ab", "Pr(>|z|)"]
    
    # c path: Ab -> Group, controlling for Age and Sex.
    res_row$c_path_est <- sum_total["Ab", "Estimate"]
    res_row$c_path_p   <- sum_total["Ab", "Pr(>|z|)"]
    
    # 4. Bootstrap mediation analysis.
    # X = Ab
    # M = TA
    # Y = Group
    
    # Logistic path coefficients are in log-odds units.
    # ACME, ADE, and total effects are probability differences, not a*b or c.
    # Compare the exposure's 25th and 75th percentiles in this pair's sample.
    exposure_contrast <- unname(quantile(df_temp$Ab, c(0.25, 0.75)))
    res_row$Exposure_Control_Value <- exposure_contrast[1]
    res_row$Exposure_Treat_Value <- exposure_contrast[2]
    med.out <- mediate(
      med.fit,
      out.fit,
      treat = "Ab",
      mediator = "TA",
      control.value = exposure_contrast[1],
      treat.value = exposure_contrast[2],
      boot = TRUE,
      sims = 1000
    )
    
    res_row$ACME_Estimate <- med.out$d.avg
    res_row$ACME_CI_Lower <- med.out$d.avg.ci[1]
    res_row$ACME_CI_Upper <- med.out$d.avg.ci[2]
    res_row$ACME_p_value  <- med.out$d.avg.p
    
    res_row$ADE_Estimate <- med.out$z.avg
    res_row$ADE_CI_Lower <- med.out$z.avg.ci[1]
    res_row$ADE_CI_Upper <- med.out$z.avg.ci[2]
    res_row$ADE_p_value  <- med.out$z.avg.p
    
    res_row$Total_Estimate <- med.out$tau.coef
    res_row$Total_CI_Lower <- med.out$tau.ci[1]
    res_row$Total_CI_Upper <- med.out$tau.ci[2]
    res_row$Total_p_value  <- med.out$tau.p
    
    res_row$Prop_Estimate <- med.out$n.avg
    res_row$Prop_CI_Lower <- med.out$n.avg.ci[1]
    res_row$Prop_CI_Upper <- med.out$n.avg.ci[2]
    res_row$Prop_p_value  <- med.out$n.avg.p
    
  }, silent = TRUE)
  
  # Record model errors in Error_Message.
  
  if (inherits(try_out, "try-error")) {
    res_row$Error_Message <- as.character(try_out)
  }
  
  return(res_row)
}

stopCluster(cl)

# 6. Save results.

output_file <- "/Users/linqing/Desktop/Projects/2024 structure_FC/RMN network/Representation mapping network/6.Disease/DK68/mediation1/AD_Mediation_Analysis_Results_AbX_TAM_GroupY.csv"

write.csv(
  results_df,
  output_file,
  row.names = FALSE
)

cat(
  "All calculations completed. Results saved to:",
  output_file,
  "\n"
)



# ==============================================================================
# Part 4: TA as exposure (X), Ab as mediator (M), and Group as outcome (Y).
#
# Use logistic outcome regression and extract path estimates and p-values.
# a-path p-values use t-tests; b-, c-, and c-prime-path p-values use z-tests.
# ==============================================================================

rm(list=ls())

library(mediation)
library(tidyverse)
library(doRNG)
library(doParallel)

set.seed(2026)

# 1. Load required packages.

packages <- c("R.matlab", "mediation", "foreach", "doParallel")

for (p in packages) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}

# 2. Load data.

mat_data <- readMat(
  "/Users/linqing/Desktop/Projects/2024 structure_FC/RMN network/Representation mapping network/6.Disease/DK68/mediation/AD_data.mat" #replace your path
)

dat_harmonized <- mat_data$dat.harmonized   # Dimensions: n_regions x n_terms x n_subjects.
Ab_data <- mat_data$Ab.data                 # Dimensions: n_subjects x n_regions.
age <- as.numeric(mat_data$age)              # Length: n_subjects.
sex <- as.numeric(mat_data$sex)              # Length: n_subjects.
disease <- as.numeric(mat_data$disease)      # Length: n_subjects; outcome Y.

# Group coding.
# Ensure Group is coded as 0/1.

if (!all(disease %in% c(0, 1))) {
  
  disease <- ifelse(
    disease == min(disease, na.rm = TRUE),
    0,
    1
  )
  
  cat(
    "Note: disease has been recoded as 0/1 for logistic regression.\n"
  )
}

# Load region and term labels.

region_info <- read.csv(
  "/Users/linqing/Desktop/Projects/2024 structure_FC/RMN network/Representation mapping network/6.Disease/DK68/DK68_name.csv",
  stringsAsFactors = FALSE
)

term_info <- read.csv(
  "/Users/linqing/Desktop/Projects/2025_RMN/data/Term.csv",
  stringsAsFactors = FALSE
)

region_names <- region_info$label
term_names <- term_info$Term


# 3. Configure the parallel backend.

num_cores <- detectCores() - 1

cl <- makeCluster(num_cores)

registerDoParallel(cl)

cat(
  "Parallel processing enabled. Number of cores:",
  num_cores,
  "\n"
)


# ==============================================================================
# 4. Load the selected region-term pairs.
# ==============================================================================

pair_index_file <- 
  "/Users/linqing/Desktop/Projects/2024 structure_FC/RMN network/Representation mapping network/6.Disease/DK68/mediation/AD_mediation_index.csv"

pair_df <- read.csv(
  pair_index_file,
  header = TRUE,
  stringsAsFactors = FALSE
)

# Use the first column as the region index and the second as the term index (R uses 1-based indices).

task_grid <- data.frame(
  region_index = as.numeric(pair_df[, 1]),
  term_index   = as.numeric(pair_df[, 2])
)

total_tasks <- nrow(task_grid)

cat(
  "Number of mediation models to run:",
  total_tasks,
  "\n"
)


# ==============================================================================
# 5. Run models in parallel.
# ==============================================================================

results_df <- foreach(
  k = 1:total_tasks,
  .combine = rbind,
  .packages = c("mediation")
) %dorng% {
  
  i <- task_grid$region_index[k]
  j <- task_grid$term_index[k]
  
  TA_val <- dat_harmonized[i, j, ]
  Ab_val <- Ab_data[, i]
  
  # ============================================================================
  # Build the analysis data.
  #
  # X = TA
  # M = Ab
  # Y = Group
  # ============================================================================
  
  df_temp <- data.frame(
    TA = TA_val,        # X: exposure.
    Ab = Ab_val,        # M: mediator.
    Group = disease,    # Y: outcome.
    Age = age,
    Sex = sex
  )
  
  res_row <- data.frame(
    
    region_index = i,
    region = region_names[i],
    term_index = j,
    term = term_names[j],
    
    # --------------------------------------------------------------------------
    # Path coefficients a, b, c, and c-prime and their p-values.
    # --------------------------------------------------------------------------
    
    a_path_est = NA_real_,
    a_path_p = NA_real_,
    
    b_path_est = NA_real_,
    b_path_p = NA_real_,
    
    c_prime_path_est = NA_real_,
    c_prime_path_p = NA_real_,
    
    c_path_est = NA_real_,
    c_path_p = NA_real_,
    
    # --------------------------------------------------------------------------
    # Bootstrap mediation estimates.
    # --------------------------------------------------------------------------
    
    ACME_Estimate = NA_real_,
    ACME_CI_Lower = NA_real_,
    ACME_CI_Upper = NA_real_,
    ACME_p_value = NA_real_,
    
    ADE_Estimate = NA_real_,
    ADE_CI_Lower = NA_real_,
    ADE_CI_Upper = NA_real_,
    ADE_p_value = NA_real_,
    
    Total_Estimate = NA_real_,
    Total_CI_Lower = NA_real_,
    Total_CI_Upper = NA_real_,
    Total_p_value = NA_real_,
    
    Prop_Estimate = NA_real_,
    Prop_CI_Lower = NA_real_,
    Prop_CI_Upper = NA_real_,
    Prop_p_value = NA_real_,
    
    Error_Message = "",
    
    stringsAsFactors = FALSE
  )
  
  
  # Capture model errors using try.
  
  try_out <- try({
    
    # ========================================================================
    # 1. Mediator regression (a path).
    #
    # TA -> Ab
    #
    # X = TA
    # M = Ab
    # ========================================================================
    
    med.fit <- lm(
      Ab ~ TA + Age + Sex,
      data = df_temp
    )
    
    
    # ========================================================================
    # 2. Outcome regression (b and c-prime paths).
    #
    # Ab -> Group | TA
    # TA -> Group | Ab
    #
    # Y = Group
    # ========================================================================
    
    out.fit <- glm(
      Group ~ Ab + TA + Age + Sex,
      data = df_temp,
      family = binomial(link = "logit")
    )
    
    
    # ========================================================================
    # 3. Total-effect regression (c path).
    #
    # TA -> Group
    #
    # X = TA
    # Y = Group
    # ========================================================================
    
    total.fit <- glm(
      Group ~ TA + Age + Sex,
      data = df_temp,
      family = binomial(link = "logit")
    )
    
    
    # ========================================================================
    # Extract path coefficients a, b, c, and c-prime.
    # ========================================================================
    
    sum_med <- summary(med.fit)$coefficients
    sum_out <- summary(out.fit)$coefficients
    sum_total <- summary(total.fit)$coefficients
    
    
    # ------------------------------------------------------------------------
    # a path: TA -> Ab.
    # ------------------------------------------------------------------------
    
    res_row$a_path_est <- sum_med["TA", "Estimate"]
    res_row$a_path_p   <- sum_med["TA", "Pr(>|t|)"]
    
    
    # ------------------------------------------------------------------------
    # b path: Ab -> Group, controlling for TA, Age, and Sex.
    # ------------------------------------------------------------------------
    
    res_row$b_path_est <- sum_out["Ab", "Estimate"]
    res_row$b_path_p   <- sum_out["Ab", "Pr(>|z|)"]
    
    
    # ------------------------------------------------------------------------
    # c-prime path: TA -> Group, controlling for Ab, Age, and Sex.
    # ------------------------------------------------------------------------
    
    res_row$c_prime_path_est <- sum_out["TA", "Estimate"]
    res_row$c_prime_path_p   <- sum_out["TA", "Pr(>|z|)"]
    
    
    # ------------------------------------------------------------------------
    # c path: TA -> Group, controlling for Age and Sex.
    # ------------------------------------------------------------------------
    
    res_row$c_path_est <- sum_total["TA", "Estimate"]
    res_row$c_path_p   <- sum_total["TA", "Pr(>|z|)"]
    
    
    # ========================================================================
    # 4. Bootstrap mediation analysis.
    #
    # X = TA
    # M = Ab
    # Y = Group
    # ========================================================================
    
    # Logistic path coefficients are in log-odds units.
    # ACME, ADE, and total effects are probability differences, not a*b or c.
    # Compare the exposure's 25th and 75th percentiles in this pair's sample.
    exposure_contrast <- unname(quantile(df_temp$TA, c(0.25, 0.75)))
    res_row$Exposure_Control_Value <- exposure_contrast[1]
    res_row$Exposure_Treat_Value <- exposure_contrast[2]
    med.out <- mediate(
      med.fit,
      out.fit,
      treat = "TA",
      mediator = "Ab",
      control.value = exposure_contrast[1],
      treat.value = exposure_contrast[2],
      boot = TRUE,
      sims = 1000
    )
    
    
    # ------------------------------------------------------------------------
    # ACME: average indirect effect on the probability-difference scale.
    # TA -> Ab -> Group
    # ------------------------------------------------------------------------
    
    res_row$ACME_Estimate <- med.out$d.avg
    res_row$ACME_CI_Lower <- med.out$d.avg.ci[1]
    res_row$ACME_CI_Upper <- med.out$d.avg.ci[2]
    res_row$ACME_p_value  <- med.out$d.avg.p
    
    
    # ------------------------------------------------------------------------
    # ADE: average direct effect on the probability-difference scale.
    # TA -> Group
    # Controlling for Ab.
    # ------------------------------------------------------------------------
    
    res_row$ADE_Estimate <- med.out$z.avg
    res_row$ADE_CI_Lower <- med.out$z.avg.ci[1]
    res_row$ADE_CI_Upper <- med.out$z.avg.ci[2]
    res_row$ADE_p_value  <- med.out$z.avg.p
    
    
    # ------------------------------------------------------------------------
    # Total effect on the probability-difference scale.
    # TA -> Group
    # ------------------------------------------------------------------------
    
    res_row$Total_Estimate <- med.out$tau.coef
    res_row$Total_CI_Lower <- med.out$tau.ci[1]
    res_row$Total_CI_Upper <- med.out$tau.ci[2]
    res_row$Total_p_value  <- med.out$tau.p
    
    
    # ------------------------------------------------------------------------
    # Proportion Mediated
    # ------------------------------------------------------------------------
    
    res_row$Prop_Estimate <- med.out$n.avg
    res_row$Prop_CI_Lower <- med.out$n.avg.ci[1]
    res_row$Prop_CI_Upper <- med.out$n.avg.ci[2]
    res_row$Prop_p_value  <- med.out$n.avg.p
    
  }, silent = TRUE)
  
  
  # ==========================================================================
  # Record model errors in Error_Message.
  # ==========================================================================
  
  if (inherits(try_out, "try-error")) {
    res_row$Error_Message <- as.character(try_out)
  }
  
  return(res_row)
}


# ==============================================================================
# 6. Stop the parallel cluster.
# ==============================================================================

stopCluster(cl)


# ==============================================================================
# 7. Save results.
# ==============================================================================

output_file <- 
  "/Users/linqing/Desktop/Projects/2024 structure_FC/RMN network/Representation mapping network/6.Disease/DK68/mediation1/AD_Mediation_Analysis_Results_TAX_AbM_GroupY.csv"

write.csv(
  results_df,
  output_file,
  row.names = FALSE
)



cat(
  "All calculations completed. Results saved to:",
  output_file,
  "\n"
)



