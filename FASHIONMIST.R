library(keras)
library(reticulate)
use_condaenv("tf", required = TRUE)


# Load Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
train_images <- fashion_mnist$train$x
train_labels <- fashion_mnist$train$y
test_images <- fashion_mnist$test$x
test_labels <- fashion_mnist$test$y

# Preprocess the data
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))
test_images <- test_images / 255

# Define the CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Train the model
history <- model %>% fit(
  train_images, train_labels,
  epochs = 10,
  batch_size = 64,
  validation_split = 0.2
)

# Evaluate the model
eval_result <- model %>% evaluate(test_images, test_labels)
cat("Test accuracy:", eval_result[[2]], "\n")


predictions <- model %>% predict(test_images)


# Get the predicted classes
predicted_classes <- apply(predictions, 1, which.max)



# Print predicted classes for the first two images
cat("Predicted classes for the first image:", predicted_classes[1], "\n")
cat("Predicted classes for the second image:", predicted_classes[2], "\n")

