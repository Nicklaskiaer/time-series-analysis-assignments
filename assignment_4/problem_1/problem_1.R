# Assignment 4, Problem 1
# This script is meant to contain the code for tasks 1.1--1.5.

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


# ------------------------------------------------------------------------------------
# 1.2

# Set seed again for reproducibility in this task
set.seed(42)

# Observation noise standard deviation
sigma2 <- 1

# Vectors to store the hidden state and noisy observations
X_12 <- numeric(n + 1)
Y_12 <- numeric(n + 1)

# Set initial value
X_12[1] <- X0

# The first observation is the initial state plus observation noise
Y_12[1] <- X_12[1] + rnorm(1, mean = 0, sd = sigma2)

# Simulate one hidden state trajectory and the corresponding observations
for (t in 2:(n + 1)) {
  e1_t <- rnorm(1, mean = 0, sd = sigma1)
  e2_t <- rnorm(1, mean = 0, sd = sigma2)

  X_12[t] <- a * X_12[t - 1] + b + e1_t
  Y_12[t] <- X_12[t] + e2_t
}

# Save plot as PNG in the same folder as the plot from task 1.1
png("assignment_4/problem_1/figures/trajectory_1_2_plot.png", width = 1200, height = 600)

plot(
  time, X_12,
  type = "l",
  col = "blue",
  lwd = 2,
  xlab = "Time",
  ylab = "Value",
  main = "Hidden state and noisy observations",
  ylim = range(c(X_12, Y_12))
)

points(
  time, Y_12,
  col = "red",
  pch = 16,
  cex = 0.7
)

legend(
  "bottomright",
  legend = c("Hidden state X_t", "Noisy observations Y_t"),
  col = c("blue", "red"),
  lty = c(1, NA),
  pch = c(NA, 16),
  lwd = c(2, NA),
  bty = "n"
)

# Close the file
dev.off()


# ------------------------------------------------------------------------------------
# 1.3

# Code for task 1.3 will be added here.


# ------------------------------------------------------------------------------------
# 1.4

# Code for task 1.4 will be added here.


# ------------------------------------------------------------------------------------
# 1.5

# Code for task 1.5 will be added here.
