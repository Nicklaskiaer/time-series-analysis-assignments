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

# Set seed for reproducibility
# The Kalman filter itself is deterministic, but we keep the seed for consistency
set.seed(42)

# Import implemented Kalman filter function
source("assignment_4/problem_1/kalmanFilter_implemented.R")

# Define parameter vector theta = (a, b, sigma1)
theta <- c(a, b, sigma1)

# Observation noise variance
# Since sigma2 = 1, the variance is sigma2^2 = 1
R <- sigma2^2

# Initial prior mean and variance for the Kalman filter
# We use the same initial value as in the simulation
x_prior <- X0
P_prior <- 10

# Apply the Kalman filter to the observations from Question 1.2
kf_13 <- myKalmanFilter(
  y = Y_12,
  theta = theta,
  R = R,
  x_prior = x_prior,
  P_prior = P_prior
)

# ------------------------------------------------------------
# Compute the predicted state X_hat_{t+1|t}
# ------------------------------------------------------------

# The Kalman filter returns the filtered estimate X_hat_{t|t}.
# To obtain the one-step-ahead prediction X_hat_{t+1|t},
# we propagate the filtered estimate through the state equation:
# X_hat_{t+1|t} = a * X_hat_{t|t} + b
x_pred_1step <- a * kf_13$x_filt + b

# The corresponding prediction variance is:
# Sigma^{xx}_{t+1|t} = a^2 * Sigma^{xx}_{t|t} + sigma1^2
P_pred_1step <- a^2 * kf_13$P_filt + sigma1^2

# The prediction X_hat_{t+1|t} is a prediction for the next time point.
# Since the data are observed from time 0 to 100, we only keep predictions
# for time 1 to 100 and remove the final prediction for time 101.
x_pred_1step <- x_pred_1step[1:n]
P_pred_1step <- P_pred_1step[1:n]

# Time points corresponding to X_hat_{t+1|t}
time_pred <- 1:n

# Compute 95% confidence interval around X_hat_{t+1|t}
ci_upper <- x_pred_1step + 1.96 * sqrt(P_pred_1step)
ci_lower <- x_pred_1step - 1.96 * sqrt(P_pred_1step)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

# Save plot as PNG
png("assignment_4/problem_1/figures/question_1_3_plot.png", width = 1200, height = 600)

# Plot the true latent state X_t
plot(
  time, X_12,
  type = "l",
  col = "blue",
  lwd = 2,
  xlab = "Time",
  ylab = "Value",
  main = "Kalman filter one-step-ahead prediction",
  ylim = range(c(X_12, Y_12, x_pred_1step, ci_lower, ci_upper))
)

# Add noisy observations Y_t
points(
  time, Y_12,
  col = "red",
  pch = 16,
  cex = 0.6
)

# Add predicted state X_hat_{t+1|t}
lines(
  time_pred,
  x_pred_1step,
  col = "black",
  lwd = 2,
  lty = 2
)

# Add 95% confidence interval around the predicted state
lines(
  time_pred,
  ci_upper,
  col = "darkgray",
  lwd = 1,
  lty = 3
)

lines(
  time_pred,
  ci_lower,
  col = "darkgray",
  lwd = 1,
  lty = 3
)

# Add legend
legend(
  "bottomright",
  legend = c(
    expression("Latent state " * X[t]),
    expression("Noisy observations " * Y[t]),
    expression("Predicted state " * hat(X)[t+1|t]),
    "95% confidence interval"
  ),
  col = c("blue", "red", "black", "darkgray"),
  lty = c(1, NA, 2, 3),
  pch = c(NA, 16, NA, NA),
  lwd = c(2, NA, 2, 1),
  bty = "n"
)

# Close the PNG file
dev.off()