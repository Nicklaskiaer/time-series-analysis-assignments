# Final plots for assignment 3, problem 1.
#
# Model used in the assignment:
#   X_t + phi1 * X_{t-1} + phi2 * X_{t-2} = epsilon_t
#
# arima.sim() uses the standard R convention:
#   X_t = ar1 * X_{t-1} + ar2 * X_{t-2} + epsilon_t
# so the AR coefficients are c(-phi1, -phi2).

set.seed(123)

n <- 200
n_sim <- 5
lag_max <- 30

out_dir <- "assignment_3"

simulate_ar2_arima <- function(phi1, phi2, n, n_sim) {
  replicate(
    n_sim,
    arima.sim(n = n, model = list(ar = c(-phi1, -phi2)))
  )
}

simulate_ar2_recursive <- function(phi1, phi2, n, n_sim) {
  sims <- matrix(NA_real_, nrow = n, ncol = n_sim)

  for (j in seq_len(n_sim)) {
    eps <- rnorm(n)
    x <- numeric(n)

    x[1] <- eps[1]
    x[2] <- -phi1 * x[1] + eps[2]

    for (t in 3:n) {
      x[t] <- -phi1 * x[t - 1] - phi2 * x[t - 2] + eps[t]
    }

    sims[, j] <- x
  }

  sims
}

simulate_ar2 <- function(phi1, phi2, n, n_sim, recursive = FALSE) {
  if (recursive) {
    simulate_ar2_recursive(phi1, phi2, n, n_sim)
  } else {
    simulate_ar2_arima(phi1, phi2, n, n_sim)
  }
}

empirical_acfs <- function(sim_data, lag_max) {
  apply(sim_data, 2, function(x) {
    as.numeric(acf(x, plot = FALSE, lag.max = lag_max)$acf)
  })
}

theoretical_acf <- function(phi1, phi2, lag_max) {
  tryCatch(
    ARMAacf(ar = c(-phi1, -phi2), lag.max = lag_max),
    error = function(e) {
      ar1 <- -phi1
      ar2 <- -phi2
      rho <- numeric(lag_max + 1)

      rho[1] <- 1
      rho[2] <- ar1 / (1 - ar2)

      for (k in 3:(lag_max + 1)) {
        rho[k] <- ar1 * rho[k - 1] + ar2 * rho[k - 2]
      }

      rho
    }
  )
}

plot_time_series <- function(sim_data, main) {
  matplot(
    sim_data,
    type = "l",
    lty = 1,
    xlab = "Time",
    ylab = expression(X[t]),
    main = main
  )

  legend(
    "topright",
    legend = paste("Realization", seq_len(ncol(sim_data))),
    lty = 1,
    col = seq_len(ncol(sim_data)),
    bty = "n",
    cex = 0.8
  )
}

plot_acf_comparison <- function(sim_data, phi1, phi2, main) {
  acf_emp <- empirical_acfs(sim_data, lag_max)
  acf_theo <- theoretical_acf(phi1, phi2, lag_max)

  ylim_values <- acf_emp
  if (!is.null(acf_theo)) {
    ylim_values <- c(ylim_values, acf_theo)
  }

  y_range <- range(ylim_values)
  y_padding <- diff(y_range) * 0.05
  if (y_padding == 0) {
    y_padding <- 0.05
  }

  plot(
    0:lag_max,
    acf_emp[, 1],
    type = "l",
    lty = 1,
    col = 1,
    ylim = y_range + c(-y_padding, y_padding),
    xlab = "Lag",
    ylab = "ACF",
    main = main
  )

  for (i in 2:ncol(acf_emp)) {
    lines(0:lag_max, acf_emp[, i], col = i, lty = 1)
  }

  legend_text <- paste("Empirical", seq_len(ncol(acf_emp)))
  legend_col <- seq_len(ncol(acf_emp))
  legend_lwd <- rep(1, ncol(acf_emp))

  if (!is.null(acf_theo)) {
    lines(0:lag_max, acf_theo, col = "black", lwd = 3)
    legend_text <- c(legend_text, "Theoretical")
    legend_col <- c(legend_col, "black")
    legend_lwd <- c(legend_lwd, 3)
  }

  legend(
    "topright",
    legend = legend_text,
    col = legend_col,
    lty = 1,
    lwd = legend_lwd,
    bty = "n",
    cex = 0.75
  )
}

plot_combined_question <- function(question, phi1, phi2, recursive = FALSE) {
  sim_data <- simulate_ar2(phi1, phi2, n, n_sim, recursive = recursive)

  png(
    file.path(out_dir, paste0("problem1_final_", question, ".png")),
    width = 1200,
    height = 1000
  )

  par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))

  title_text <- paste0(
    "Time series plot of the AR(2) process with phi1 = ",
    phi1,
    " and phi2 = ",
    phi2
  )

  plot_time_series(sim_data, main = title_text)
  plot_acf_comparison(sim_data, phi1, phi2, main = "Empirical and theoretical ACF")

  dev.off()
}

# 1.1: same structure as in problem1.R - simulate and plot 5 realizations.
phi1 <- -0.6
phi2 <- 0.5
sim_data <- simulate_ar2_arima(phi1, phi2, n, n_sim)

png(file.path(out_dir, "problem1_final_11_timeseries.png"), width = 1200, height = 600)
plot_time_series(
  sim_data,
  main = paste0(
    "Time series plot of the AR(2) process with phi1 = ",
    phi1,
    " and phi2 = ",
    phi2
  )
)
dev.off()

# 1.2: same structure as in problem1.R - empirical ACFs with theoretical ACF.
png(file.path(out_dir, "problem1_final_12_acf.png"), width = 1200, height = 600)
plot_acf_comparison(
  sim_data,
  phi1,
  phi2,
  main = "Problem 1.2: empirical vs theoretical ACF"
)
dev.off()

# 1.3-1.6: one combined figure per coefficient set.
plot_combined_question("13", phi1 = -0.6, phi2 = -0.3)
plot_combined_question("14", phi1 = 0.6, phi2 = -0.3)
plot_combined_question("15", phi1 = -0.7, phi2 = -0.3, recursive = TRUE)
plot_combined_question("16", phi1 = -0.75, phi2 = -0.3, recursive = TRUE)
