# Welcome to Liberate.FHE!

Liberate.FHE is an open-source Fully Homomorphic Encryption (FHE) library for bridging the gap between theory and practice with a focus on performance and accuracy.

Liberate.FHE is designed to be user-friendly while delivering robust performance, high accuracy, and a comprehensive suite of convenient APIs for developing real-world privacy-preserving applications.

Liberate.FHE is a pure Python and CUDA implementation of FHE. So, Liberate.FHE supports multi-GPU operations natively.

The main idea behind the design decisions is that non-cryptographers can use the library; it should be easily hackable and integrated with more extensive software frameworks. 

Additionally, several design decisions were made to maximize the usability of the developed software:

- Make the number of dependencies minimal.
- Make the software easily hackable.
- Set the usage of multiple GPUs as the default.
- Make the resulting library easily integrated with the pre-existing software, especially Artificial Intelligence (AI) related ones.

# Key Features

- RNS-CKKS scheme is supported.
- Python is natively supported.
- Multiple GPU acceleration is supported.
- Multiparty FHE is supported.

# Quick Start

```python
from liberate import fhe
from liberate.fhe import presets

# Generate CKKS engine with preset parameters
grade = "silver"  # logN=14
params = presets.params[grade]

engine = fhe.ckks_engine(**params, verbose=True)

# Generate Keys
sk = engine.create_secret_key()
pk = engine.create_public_key(sk)
evk = engine.create_evk(sk)

# Generate test data
m0 = engine.example(-1, 1)
m1 = engine.example(-10, 10)

# encode & encrypt data
ct0 = engine.encorypt(m0, pk)
ct1 = engine.encorypt(m1, pk, level=5)

# (a + b) * b - a
result = (m0 + m1) * m1 - m0
ct_add = engine.add(ct0, ct1)  # auto leveling
ct_mult = engine.mult(ct1, ct_add, evk)
ct_result = engine.sub(ct_mult, ct0)

# decrypt & decode data
result_decrypted = engine.decrode(ct_result, sk)
```

If you would like a detailed explanation, please refer to
the [official documentation](https://docs.desilo.ai/liberate-fhe/getting-started/quick-start).

# How to Install

### Clone this repository

```shell
git clone https://github.com/Desilo/liberate-fhe.git
cd liberate-fhe
```

### Install dependencies

```shell
poetry install
```

### Run Cuda build Script.

```shell
python setup.py install
# poetry run python setup.py install
```

### Build a python package

```shell
poetry build
```

### Install Liberate.FHE library

```shell
pip install .
# poetry run python -m pip install .
```

# Documentation

Please refer to [Liberate.FHE](https://docs.desilo.ai/liberate-fhe/api-references/docs) for detailed installation
instructions, examples, and documentation.


# Citing Liberate.FHE

```text
@Misc{Liberate_FHE,
  title={{Liberate.FHE: A New FHE Library for Bridging the Gap between Theory and Practice with a Focus on Performance and Accuracy}},
  author={DESILO},
  year={2023},
  note={\url{https://github.com/Desilo/liberate-fhe}},
}
```

# License

- Liberate.FHE is available under the *BSD 3-Clause Clear license*. If you have any questions, please contact us at contact@desilo.ai.
