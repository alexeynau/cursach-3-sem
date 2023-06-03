library(RCurl)
library(TTR)
library(zoo)
library(MASS)
# Импортирование данных ---------------------------------------------------
x <- getURL("https://raw.githubusercontent.com/alexeynau/os-elm/main/data/nyc_taxi.csv")
df <- read.csv(text = x)
data <- ts(df$value)
head(df)
# df$timestamp <- as.Date.character(df$timestamp, format = '')
# df$timestamp
# ?as.Date.character()



# Визуализация ------------------------------------------------------------

plot(
  data, 
  xlim = c(0, 1000), 
  type = 'l', 
  xlab = 'Время', 
  ylab = 'Количество пассажиров',
  main = 'Динамика колиства пассажиров такси, измеряемая каждые полчаса'
)



# Проверим данные на пропуски ---------------------------------------------
sum(is.na(data))
# Вывод: "[1] 0"
# Следовательно пропусков нет


# Сглаживание данных ------------------------------------------------------
# Используем скользящее среднее
df <- SMA(data, n = 10)
df <- df[!is.na(df)]
sum(is.na(df))
# Визуализируем результат
plot(
  data[5:length(data)], 
  xlim = c(0, 500), 
  type = 'l', 
  xlab = 'Время', 
  ylab = 'Количество пассажиров',
  lwd = 2,
  main = 'Результат применения скользящего среднего к временному ряду'
)
lines(df, xlim = c(0, 500),lwd = 2, col = 'red')
legend("topleft", legend = c("Исходные данные", "Данные после сглаживания"),
       lwd = 2, col = c("black", "red"))

# Нормализация данных -----------------------------------------------------
# Используем z-масштабирование
mean_df <- mean(df)
std_df <- sd(df)
df <- scale(df)
head(df)


# Подготовка предикторов и откликов ---------------------------------------------------

window_size = 100
X <- embed(df, window_size)
y <- df[window_size + 1:length(df)]
dim(X)
length(y[!is.na(y)])

# Реализация OS-ELM -----------------------------------------
# Объявление параметров
num_of_hidden_neurons <- 0
forgetting_factor <- 0
num_of_inputs <- 0
num_of_outputs <- 0
gamma <- 0

alpha <- 0 # Веса
bias <- 0 # Смещение

beta <- 0
P <- 0

init <- function(n_hidden, n_inputs, n_outputs, ff, gamma) {
  num_of_hidden_neurons <<- n_hidden
  forgetting_factor <<- ff
  num_of_inputs <<- n_inputs
  num_of_outputs <<- n_outputs
  gamma <<- gamma
  
  # Заполняет веса и смещение случайными значениями в интервале [-1; 1]
  alpha <<- matrix(runif(n_hidden * n_inputs, -1, 1), n_hidden, n_inputs)
  bias <<- runif(n_hidden, -1, 1)
  
  beta <<- matrix(0, n_hidden, n_outputs)
  P <<- solve(gamma*diag(1, n_hidden, n_hidden))
}

# Сигмоидальная функция активации
sigmoid <- function(features) { 
  return (1/(1 + exp(-features)))
}

# Фукнция, вычисляющая h
compute_h <- function(features) {
  return (sigmoid(features %*% t(alpha) + bias))
}

# Функция обучения модели
ELM_fit <- function(features, targets) {
  H <- compute_h(features)
  ff <- forgetting_factor
  k <- 1/ff
  P <<- (k * P - P %*% t(H) %*% ginv(ff^2 + ff * H %*% P %*% t(H)) %*% H %*% P)
  beta <<- beta + P %*% t(H) %*% (targets - H %*% beta)
}

# Функция прогнозирования
ELM_predict <- function(features) {
  H <- compute_h(features)
  prediction <- H %*% beta
  return (prediction)
}



# Обучение и прогнозирование ----------------------------------------------

predictions <- array()
targets <- y[1:length(y) - 1]

init(n_hidden = 100, n_inputs = 100, n_outputs = 1, ff = 0.996, gamma = 0.0001)

# Обучение и прогнозирование
for (i in 1:(length(y) - window_size - 1)) {
  print(i)
  ELM_fit(X[i,], y[i])
  prediction <- ELM_predict(X[i+1,])
  predictions <- append(predictions, prediction)
}


predictions <- predictions[!is.na(predictions)]
y = targets[!is.na(targets)]
y = y[1:length(y)-1]
predictions <- predictions * std_df + mean_df
targets <- y * std_df + mean_df


# Коэффициент детерминации
r2_score <- function (x, y) cor(x, y) ^ 2
r2_score(targets, predictions)

# Средняя абсолютная ошибка (MAE)
mae <- function(tars, preds){
  s <- 0
  for (i in 1:length(tars)){
    s <- abs(tars[i] - preds[i]) + s
  }
  error = s/length(targets)
}

error <- mae(targets, predictions)
error
# Линейная аппроксимация по полученным результатам ------------------------
plot(
  targets, 
  targets, 
  type = 'l',
  col = 'red', 
  xlab = 'Целевое значение', 
  ylab = 'Целевое значение', 
  main = 'Линейная аппроксимация по полученным результатам '
  )
points(targets, predictions, col= 'blue', pch = 19)
lines(targets, targets, col = 'red')
legend("bottomright", legend = c("Прогнозное значение к целевым", "Целевое значение к целевым"),
       lwd = 3, col = c("blue", "red"))

# Наложение спрогнозированных результатов на целевые ----------------------
plot(
  targets[9000:10000], 
  type = 'l', 
  col = 'blue', 
  lwd = 3, 
  ylim = c(0, 30000),
  xlab = 'Время', 
  ylab = 'Значение',
  main = 'Наложение спрогнозированных результатов на целевые'
  )
lines(predictions[9000:10000], type = 'l', col = 'orange', lwd = 3)
legend("topleft", legend = c("Целевые данные", "Спрогнозированный результат"),
       lwd = 3, col = c("blue", "orange"))


# Подбор параметров -------------------------------------------------------
hiddens = c(25, 50, 75, 100)
ffs = c(0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1)
r2_scores = array()
maes = array()

iter <- 0

for (num in hiddens){
  for (lambda in ffs) {
    # Обнуление полей
    num_of_hidden_neurons <- 0
    forgetting_factor <- 0
    num_of_inputs <- 0
    num_of_outputs <- 0
    gamma <- 0
    alpha <- 0
    bias <- 0 
    beta <- 0
    P <- 0
    
    y <- df[window_size + 1:length(df)]
    predictions <- array()
    targets <- y[1:length(y) - 1]
    
    init(n_hidden = num, n_inputs = 100, n_outputs = 1, ff = lambda, gamma = 0.0001)
    
    print(iter)
    iter <- iter + 1
    for (i in 1:(length(y) - window_size - 1)) {

      ELM_fit(X[i,], y[i])
      prediction <- ELM_predict(X[i+1,])
      predictions <- append(predictions, prediction)
    }
    
    
    predictions <- predictions[!is.na(predictions)]
    y = targets[!is.na(targets)]
    y = y[1:length(y)-1]
    predictions <- predictions * std_df + mean_df
    targets <- y * std_df + mean_df
    
    
    # Добавление показателей в историю
    r2 <- r2_score(targets, predictions)
    r2_scores <- append(r2_scores, r2)
    
    error <- mae(targets, predictions)
    maes <- append(maes, error)
  }
}

plot( 
  ffs, 
  r2_scores[2:12], 
  type = 'l', 
  xlim = c(0.99, 1), 
  ylim = c(0.84, 0.97), 
  lwd = 3, 
  ylab = 'Коэффициент дтерминации', 
  xlab = 'Фактор забывания',
  main = 'Зависимость коэффициента детерминации от количества нейронов в скрытом слое и фактора забывания'
)
lines( ffs,r2_scores[13:23], col = 'green', lwd = 3)
lines( ffs,r2_scores[24:34], col = 'red', lwd = 3)
lines( ffs, r2_scores[35:45],col = 'blue', lwd = 3)
legend(
  "bottomleft", 
  legend = c(
    "n = 25", 
    "n = 50",
    "n = 75",
    "n = 100"
    ),
  lwd = 3, 
  col = c(
    "black", 
    "green",
    "red",
    "blue"
    )
  )

plot( 
  ffs, 
  maes[2:12], 
  type = 'l', 
  ylim = c(600,2000),
  lwd = 3, 
  ylab = 'MAE', 
  xlab = 'Фактор забывания',
  main= 'Зависимость средней абсолютной ошибки от количества нейронов в скрытом слое и фактора забывания'
)
lines(ffs, maes[13:23], col = 'green', lwd = 3)
lines(ffs, maes[24:34], col = 'red', lwd = 3)
lines(ffs, maes[35:45],col = 'blue', lwd = 3)
legend(
  "bottomleft", 
  legend = c(
    "n = 25", 
    "n = 50",
    "n = 75",
    "n = 100"
  ),
  lwd = 3, 
  col = c(
    "black", 
           "green",
           "red",
           "blue"
  )
)
