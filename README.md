# Informações
A Rede Neural implementada segue a arquitetura de LeNet-5 e foi criada, a priore, para dois conjuntos de dados: MNIST e CIFAR-10. Os dados são fornecidos através da lib [sklearn](https://scikit-learn.org/stable/) pelo site [OpenML](https://www.openml.org/).

A arquitetura LeNet-5 possui três tipos de layers:

1 - Convolutional Layers (CONV);<br/>
2 - Pooling Layers (POOL);<br/>
3 - Fully-Connected Layers (FC).

As camadas na implementação estão dispostas da seguinte maneira:

1 - CNN é composta por um conjunto de 6 filtros (5×5), stride=1.<br/>
2 - POOL (2×2), stride=2, para reduzir o tamanho espacial das matrizes resultantes.<br/>
3 - CNN (5×5) com 16 filtros e stride=1.<br/>
4 - POOL (2×2), stride=2.<br/>
5 - Os mapas de características são achatados (flatten), formando 400 nós (5x5x16) para a próxima camanda FC.<br/>
6 - FC com 120 nós.<br/>
7 - FC com 84 nós.

# Testando
Para efetuar o teste, basta clonar o repositório e instalar as dependências listadas no arquivo **requirements.txt**: pip install -r requirements.txt. Em seguida, execute o arquivo principal **__init__.py**.

Inicialmente estará com o conjunto de dados MNIST, porém pode ser alterado para o conjunto CIFAR-10 modificando o valor da variável **dataset_name** para **"cifar_10"**: dataset_name = "cifar_10".

# Resultados Obtidos
## MNIST
Ao efetuarmos os treinamentos utilizando 75% dos dados para treinamento e as configurações da CNN descritas no tópico **Informações**, a evolução da acurácia ao decorrer das épocas foi a seguinte:

![mnist_784_epocas](https://user-images.githubusercontent.com/51512175/110179791-7348cb80-7de7-11eb-8cb2-49b9a314c53e.png)

Tendo como avaliação geral os seguintes resultados:

![mnist_784_avaliacao](https://user-images.githubusercontent.com/51512175/110179984-c4f15600-7de7-11eb-8887-f1b2492b4a3d.png)

## CIFAR-10
Ao efetuarmos os treinamentos utilizando 75% dos dados para treinamento e as configurações da CNN descritas no tópico **Informações**, a evolução da acurácia ao decorrer das épocas foi a seguinte:

![cifar_10_epocas](https://user-images.githubusercontent.com/51512175/110180045-de929d80-7de7-11eb-9718-efa13793f4b5.png)

Tendo como avaliação geral os seguintes resultados:

![cifar_10_avaliacao](https://user-images.githubusercontent.com/51512175/110180054-e0f4f780-7de7-11eb-857c-975bb23b7d9d.png)
