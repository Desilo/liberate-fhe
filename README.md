# Welcome to Liberate.FHE!

![Liberate Logo](https://github.com/Desilo/liberate/blob/cf11bcc3e7a768687c4469b64cf693cf424fec37/docs/images/Logo%20Liberate.FHE.png)

Liberate.FHE is an open-source Fully Homomorphic Encryption (FHE) library for bridging the gap between theory and practice with a focus on performance and accuracy. 

Despite significant advancements in fully homomorphic encryption (FHE), the utilization of modern computing resources, such as GPU acceleration, remains limited. Additionally, the accuracy of homomorphic computations often falls short of theoretical expectations. To address these shortcomings, Liberate.FHE adopts a practical engineering approach, refining existing formulations to enhance FHE performance and accuracy. Leveraging these advancements, a new FHE library is developed.

Liberate.FHE supports multi-GPU operations natively, and its Application Programming Interface (API) is designed to be simple by employing the most widely used presets. The main idea behind the design decisions is that non-cryptographers can use the library; it should be easily hackable and integrated with more extensive software frameworks. 

Additionally, several design decisions were made to maximize the usability of the developed software:

- Make the number of dependencies minimal.
- Make the software easily hackable.
- Set the usage of multiple GPUs as the default.
- Make the resulting library easily integrated with the pre-existing software, especially the Artificial Intelligence (AI) related ones.



# Key Features

- RNS-CKKS scheme is supported.
- Python is natively supported.
- Multiple GPU acceleration is supported.
- Multiparty FHE is supported.



# Quick Start

```python
from liberate import fhe
from liberate.fhe import presets


# Generate ckks engine with preseting parameter
grade = "silver" # logN=14
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
ct_add = engine.add(ct0, ct1) # auto leveling
ct_mult = engine.mult(ct1, ct_add, evk)
ct_result = engine.sub(ct_mult, ct0)

# decrypt & decode data
result_decrypted = engine.decrode(ct_result, sk)
```

If you would like a detailed explanation, please refer to the [official documentation](https://docs.desilo.ai/liberate-fhe/getting-started/quick-start).



# How to Install

### Clone this repository

```shell
git clone https://github.com/Desilo/liberate.git
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

Please refer to [Liberate.FHE](https://docs.desilo.ai/liberate-fhe/api-references/docs) for detailed installation instructions, examples, and documentation.



# Features To Be Supported

- CKKS bootstrapping
- LIBERATE.FHE CPU version



# License

- Liberate.FHE is available under the *BSD 3-Clause Clear license*.



# Support

- Support forum : TBD









