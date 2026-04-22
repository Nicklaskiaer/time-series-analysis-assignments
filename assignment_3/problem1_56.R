# Plot empirical ACFs for the simulations in assignment 1.5 and 1.6.
#
# Model:
#   X_t + phi1 * X_{t-1} + phi2 * X_{t-2} = epsilon_t
# so:
#   X_t = -phi1 * X_{t-1} - phi2 * X_{t-2} + epsilon_t
#
# The simulations are generated recursively because assignment 1.5 is on
# the stationarity boundary, where arima.sim() will fail.

set.seed(123)

n <- 200
n_sim <- 5
lag_max <- 30

simulate_ar2 <- function(phi1, phi2, n, n_sim) {
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

plot_acfs <- function(phi1, phi2, assignment, add_theoretical = FALSE) {
  sims <- simulate_ar2(phi1, phi2, n, n_sim)

  acf_empirical <- apply(sims, 2, function(x) {
    as.numeric(acf(x, plot = FALSE, lag.max = lag_max)$acf)
  })

  plot(
    0:lag_max,
    acf_empirical[, 1],
    type = "l",
    lty = 1,
    col = 1,
    ylim = range(acf_empirical),
    xlab = "Lag",
    ylab = "ACF",
    main = bquote(.(assignment) ~ ": " ~ phi[1] == .(phi1) ~ "," ~ phi[2] == .(phi2))
  )

  for (i in 2:ncol(acf_empirical)) {
    lines(0:lag_max, acf_empirical[, i], col = i, lty = 1)
  }

  legend_text <- paste("Realization", seq_len(n_sim))
  legend_col <- seq_len(n_sim)
  legend_lwd <- rep(1, n_sim)

  if (add_theoretical) {
    acf_theoretical <- ARMAacf(ar = c(-phi1, -phi2), lag.max = lag_max)
    lines(0:lag_max, acf_theoretical, col = "black", lwd = 3)

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
    cex = 0.8
  )
}

png("assignment_3/problem1_56_acf.png", width = 1200, height = 900)

par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))

plot_acfs(phi1 = -0.7, phi2 = -0.3, assignment = "Assignment 1.5")
plot_acfs(phi1 = -0.75, phi2 = -0.3, assignment = "Assignment 1.6", add_theoretical = TRUE)

dev.off()
