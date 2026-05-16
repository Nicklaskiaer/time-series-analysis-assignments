# Set seed for reproducibility
set.seed(42)

# Parameters
n <- 100
a <- 0.9
b <- 1
sigma1 <- 1   # Standard deviation of system noise
sigma2 <- 1   # Standard deviation of observation noise
X0 <- 5

# Create vectors to store the latent state X_t and observations Y_t
X <- numeric(n + 1)
Y <- numeric(n + 1)

# Set initial state
X[1] <- X0

# Simulate the latent state process X_t
# X_t is the true hidden state that we do not observe perfectly
for (t in 2:(n + 1)) {
  e1_t <- rnorm(1, mean = 0, sd = sigma1)
  X[t] <- a * X[t - 1] + b + e1_t
}

# Simulate noisy observations Y_t
# Y_t is what we observe: the true state X_t plus measurement noise
for (t in 1:(n + 1)) {
  e2_t <- rnorm(1, mean = 0, sd = sigma2)
  Y[t] <- X[t] + e2_t
}

# Time index from 0 to 100
time <- 0:n

# Save a wider plot as PNG
png("assignment_4/problem_1/figures/question_1_2_plot.png", width = 1200, height = 600)

# Plot the latent state
plot(
  time, X,
  type = "l",
  lwd = 2,
  col = "black",
  xlab = "Time",
  ylab = "Value",
  main = "Latent state and noisy observations"
)

# Add noisy observations as points
points(
  time, Y,
  pch = 16,
  col = "red"
)

# Add legend
legend(
  "bottomright",
  legend = c("Latent state X_t", "Noisy observations Y_t"),
  col = c("black", "red"),
  lty = c(1, NA),
  pch = c(NA, 16),
  lwd = c(2, NA),
  bty = "n"
)

# Close the PNG file
dev.off()