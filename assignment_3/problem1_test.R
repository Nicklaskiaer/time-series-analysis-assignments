# Plot time series simulations for the AR(2) processes in assignment 1.1-1.6.
#
# Model:
#   X_t + phi1 * X_{t-1} + phi2 * X_{t-2} = epsilon_t
# which is simulated recursively as:
#   X_t = -phi1 * X_{t-1} - phi2 * X_{t-2} + epsilon_t

set.seed(123)

n <- 200
n_sim <- 5

processes <- data.frame(
  assignment = c("1.1 / 1.2", "1.3", "1.4", "1.5", "1.6"),
  phi1 = c(-0.6, -0.6, 0.6, -0.7, -0.75),
  phi2 = c(0.5, -0.3, -0.3, -0.3, -0.3)
)

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

plot_ar2_process <- function(phi1, phi2, assignment, n, n_sim) {
  sims <- simulate_ar2(phi1, phi2, n, n_sim)

  matplot(
    sims,
    type = "l",
    lty = 1,
    xlab = "Time",
    ylab = expression(X[t]),
    main = bquote(.(assignment) ~ ": " ~ phi[1] == .(phi1) ~ "," ~ phi[2] == .(phi2))
  )

  legend(
    "topleft",
    legend = paste("Realization", seq_len(n_sim)),
    col = seq_len(n_sim),
    lty = 1,
    bty = "n",
    cex = 0.8
  )
}

out_file <- file.path("assignment_3", "problem1_test_timeseries.png")
png(out_file, width = 1400, height = 1800)

par(mfrow = c(3, 2), mar = c(4, 4, 3, 1))

for (i in seq_len(nrow(processes))) {
  plot_ar2_process(
    phi1 = processes$phi1[i],
    phi2 = processes$phi2[i],
    assignment = processes$assignment[i],
    n = n,
    n_sim = n_sim
  )
}

plot.new()
text(
  0.5,
  0.55,
  "Assignment 1.2 uses the same simulated process as 1.1.\nIt asks for ACF plots, not a new time series process.",
  cex = 1.2
)

dev.off()

message("Saved plot to: ", out_file)
