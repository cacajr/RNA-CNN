# Informações
A Rede Neural implementada segue a arquitetura de LeNet-5 e foi criada, a priore, para dois conjuntos de dados: MNIST e CIFAR-10. Os dados são fornecidos através da lib [sklearn](https://scikit-learn.org/stable/) pelo site [OpenML](https://www.openml.org/).

A arquitetura LeNet-5 possui três tipos de layers:

1 - Convolutional Layers (CONV);
2 - Pooling Layers (POOL);
3 - Fully-Connected Layers (FC).

As camadas na implementação estão dispostas da seguinte maneira:

1 - CNN é composta por um conjunto de 6 filtros (5×5), stride=1.
2 - POOL (2×2), stride=2, para reduzir o tamanho espacial das matrizes resultantes.
3 - CNN (5×5) com 16 filtros e stride=1.
4 - POOL (2×2), stride=2.
5 - Os mapas de características são achatados (flatten), formando 400 nós (5x5x16) para a próxima camanda FC.
6 - FC com 120 nós.
7 - FC com 84 nós.

# Testando
Para efetuar o teste, basta clonar o repositório e instalar as dependências listadas no arquivo **requirements.txt**: pip install -r requirements.txt. Em seguida execute, o arquivo principal **__init__.py**.

Inicialmente estará com o conjunto de dados MNIST, porém pode ser alterado para o conjunto CIFAR-10 modificando o valor da variável **dataset_name** para **"cifar_10"**: dataset_name = "cifar_10".