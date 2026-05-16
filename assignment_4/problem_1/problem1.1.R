# Set seed for reproducibility
set.seed(42)

# Parameters
n <- 100
n_sim <- 5
a <- 0.9
b <- 1
sigma1 <- 1
X0 <- 5

# Matrix to store simulations
X <- matrix(NA, nrow = n + 1, ncol = n_sim)

# Set initial value
X[1, ] <- X0

# Simulate 5 independent trajectories
for (j in 1:n_sim) {
  for (t in 2:(n + 1)) {
    e_t <- rnorm(1, mean = 0, sd = sigma1)
    X[t, j] <- a * X[t - 1, j] + b + e_t
  }
}

# Time index from 0 to 100
time <- 0:n

# Save a wider plot as PNG
png("assignment_4/problem_1/figures/question_1_1_plot.png", width = 1200, height = 600)

matplot(
  time, X,
  type = "l",
  lty = 1,
  lwd = 2,
  xlab = "Time",
  ylab = expression(X[t]),
  main = "Five independent realizations of the state process"
)

legend(
  "bottomright",
  legend = paste("Simulation", 1:n_sim),
  col = 1:n_sim,
  lty = 1,
  lwd = 2,
  bty = "n"
)

# Close the file
dev.off()