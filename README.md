# [Transformer-Network-for-Short-term-Photovoltaic-Power-Generation-Forecasting] [Link to Published Paper](https://d1wqtxts1xzle7.cloudfront.net/105220599/TSP_CSSE_38514-libre.pdf?1692776424=&response-content-disposition=inline%3B+filename%3DCT_NET_A_Novel_Convolutional_Transformer.pdf&Expires=1704184273&Signature=DrJ2~YV6Fm6aTtcI9Yo5nLZJ4d1eXcKw9PMl6CqRSzg9WeOr4Ea8QXLp8JpEeKbdzERWF391QqYQ1XJQHkYoClq~QDSSwU5PWESEfcc7r5vyA9GMjqP0-bgB2gnDjOp2fpz6R2H6mgEQllBLG7w7YsfDB0ImqgLaxcveyKX6Ib1fmVm-6rT-QQbxvKonsnhIKUKw~r6s4gBq5m8s1h7ybWb5T8alAxhOx2kFsRjfPTLdckizb~YslvSvwdys5BHfEfTKCcZlwqXVQMmoI-jy4iLtl7l-6LAICd2L0QYIzN6frA5TsTtQtPx2-CKtioYvZKcsX7G9GAaHrmwlvFlUmg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
The number of PV systems will increase rapidly in the future due to the policies of the government and international organizations, and the advantages of PV technology. However, the variability of PV power generation creates different negative impacts on the electric grid system, such as the stability, reliability, and planning of the operation, aside from the economic benefits. Therefore, accurate forecasting of PV power generation is significantly important to stabilize and secure grid operation and promote large-scale PV power integration. Current attempts in PV forecasting are usually prone due to environmental conditions that directly decrease the performance of the energy distribution system. To handle these issues, we propose an efficient convolutional-based transformer framework (CTN-PV) for accurate PV power forecasting comprised of three main modules: Initially, the acquired PV generation data is forward to the preprocessing for data refinement. Next, for the data encoding and decoding CNN-multi-head attention (CNN-MHA) and MHA are developed, where the encoder module is mainly composed of 1D convolutional and MHA layers that extract local and contextual features, while the decoder part is composed of MHA and feed-forward layers to generate the final output prediction. Finally, to verify the strength and stability of the model standard error metrics are considered such as mean squared error (MSE), root mean squared error (RMSE), and mean absolute percentage error (MAPE). An ablation study and comparative analysis with competitive state-of-the-art revealed the effectiveness of our proposed strategy over publicly available benchmark data. In addition, the proposed model is investigated in terms of its complexity and size for the possible deployment in the smart grid. 


## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install absl-py==0.15.0 alabaster==0.7.12 ... (add all other dependencies)
```

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/Transformer-Network-for-Short-term-Photovoltaic-Power-Generation-Forecasting.git
cd Transformer-Network-for-Short-term-Photovoltaic-Power-Generation-Forecasting
pip install -r requirements.txt
```

## Usage

1. Acquire PV generation data.
2. Run data preprocessing.
3. Train and evaluate the CTN-PV model.
4. Analyze results using standard error metrics.

Example usage:

```python
# Import necessary modules
import your_module

# Load and preprocess PV generation data
data = your_module.load_data('path/to/data.csv')
preprocessed_data = your_module.preprocess_data(data)

# Train the CTN-PV model
your_model = your_module.train_model(preprocessed_data)

# Evaluate the model
results = your_module.evaluate_model(your_model, preprocessed_data)

# Print the results
print(results)
```

## Contributing

We welcome contributions to enhance the capabilities and performance of the CTN-PV model. If you'd like to contribute, please follow our [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We acknowledge the contributions and support from the open-source community and the developers of the libraries and tools used in this project. Special thanks to [List of contributors] for their valuable input.

--- 

Feel free to customize this template further based on your project's specific details and requirements.
