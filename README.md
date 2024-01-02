# [CT-Net: A Novel Convolutional Transformer-Based Network for Short-Term Solar Energy Forecasting Using Climatic Information ](https://d1wqtxts1xzle7.cloudfront.net/105220599/TSP_CSSE_38514-libre.pdf?1692776424=&response-content-disposition=inline%3B+filename%3DCT_NET_A_Novel_Convolutional_Transformer.pdf&Expires=1704184273&Signature=DrJ2~YV6Fm6aTtcI9Yo5nLZJ4d1eXcKw9PMl6CqRSzg9WeOr4Ea8QXLp8JpEeKbdzERWF391QqYQ1XJQHkYoClq~QDSSwU5PWESEfcc7r5vyA9GMjqP0-bgB2gnDjOp2fpz6R2H6mgEQllBLG7w7YsfDB0ImqgLaxcveyKX6Ib1fmVm-6rT-QQbxvKonsnhIKUKw~r6s4gBq5m8s1h7ybWb5T8alAxhOx2kFsRjfPTLdckizb~YslvSvwdys5BHfEfTKCcZlwqXVQMmoI-jy4iLtl7l-6LAICd2L0QYIzN6frA5TsTtQtPx2-CKtioYvZKcsX7G9GAaHrmwlvFlUmg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
The number of PV systems will increase rapidly in the future due to the policies of the government and international organizations, and the advantages of PV technology. However, the variability of PV power generation creates different negative impacts on the electric grid system, such as the stability, reliability, and planning of the operation, aside from the economic benefits. Therefore, accurate forecasting of PV power generation is significantly important to stabilize and secure grid operation and promote large-scale PV power integration. Current attempts in PV forecasting are usually prone due to environmental conditions that directly decrease the performance of the energy distribution system. To handle these issues, we propose an efficient convolutional-based transformer framework for accurate PV power forecasting comprised of three main modules: Initially, the acquired PV generation data is forwarded to the preprocessing for data refinement. Next, for the data encoding and decoding CNN-multi-head attention and MHA are developed, where the encoder module is mainly composed of 1D convolutional and MHA layers that extract local and contextual features, while the decoder part is composed of MHA and feed-forward layers to generate the final output prediction. Finally, to verify the strength and stability of the model standard error metrics are considered such as mean squared error, root mean squared error, and mean absolute percentage error. An ablation study and comparative analysis with competitive state-of-the-art revealed the effectiveness of our proposed strategy over publicly available benchmark data. In addition, the proposed model is investigated in terms of its complexity and size for the possible deployment in the smart grid. 


## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install tensorflow-gpu==2.4.0 and tensorflow-estimator==2.4.0 
```
### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/Transformer-Network-for-Short-term-Photovoltaic-Power-Generation-Forecasting.git](https://github.com/munsif200/Transformer-Network-for-Short-term-Photovoltaic-Power-Generation-Forecasting-.git
cd Transformer-Network-for-Short-term-Photovoltaic-Power-Generation-Forecasting
pip install -r requirements.txt
```

## Usage

1. Acquire PV generation data of [2 eco-Kinetics, 26.5kW, mono-Si, Dual, 2010](https://dkasolarcentre.com.au/source/alice-springs/dka-m11-3-phase)
2. Run data preprocessing.
3. Train and evaluate the CTNet-PV model.
4. Analyze results using standard error metrics.

## Citations

If you find this work helpful, please consider citing it:

[Author 1], [Author 2], et al., "Transformer Network for Short-term Photovoltaic Power Generation Forecasting," Journal of Renewable Energy, 2023.

BibTeX:

```bibtex
@article{munsif2023ct,
  title={CT-NET: A Novel Convolutional Transformer-Based Network for Short-Term Solar Energy Forecasting Using Climatic Information.},
  author={Munsif, Muhammad and Ullah, Min and Fath, U and Khan, Samee Ullah and Khan, Noman and Baik, Sung Wook},
  journal={Computer Systems Science \& Engineering},
  volume={47},
  number={2},
  year={2023}
}
}
