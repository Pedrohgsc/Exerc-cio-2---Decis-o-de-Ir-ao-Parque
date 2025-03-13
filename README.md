
# Exercício 2 - Decisão de Ir ao Parque
# Dados de treinamento
X_parque = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

y_parque = np.array([0, 1, 1, 1, 0, 0, 0, 0])

# Treinamento do Perceptron
perceptron_parque = Perceptron(max_iter=1000, tol=1e-3)
perceptron_parque.fit(X_parque, y_parque)

# Teste
teste_parque = np.array([[1, 1, 0]])  # Exemplo de entrada
print("Decisão para ir ao parque:", perceptron_parque.predict(teste_parque)[0])

# Exercício 3 - Decisão sobre Comer Fora ou Cozinhar em Casa
X_comida = np.array([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 0, 1, 0],
    [1, 1, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 0, 1],
    [0, 0, 0, 1]
])

y_comida = np.array([0, 1, 0, 1, 1, 0, 0, 0])

# Treinamento do Perceptron
perceptron_comida = Perceptron(max_iter=1000, tol=1e-3)
perceptron_comida.fit(X_comida, y_comida)

# Teste
teste_comida = np.array([[1, 1, 1, 0]])  # Exemplo de entrada
print("Decisão para comer fora:", perceptron_comida.predict(teste_comida)[0])
