
<!-- PROJECT LOGO -->
<br />


  <h3 align="center">Unsupervised Method for Idiomatic Expression Subtitution (BART-UCD)</h3>

  <p align="center">
    An unsupervised method that converts sentences idiomatic expressions to their counter
    <br />




<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
      <ul>
        <li><a href="#data">Data</a></li>
        <li><a href="#training">Training</a></li>
        <li><a href="#inference">Inference</a></li>
      </ul>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
    <li><a href="#filestructure">File Structure</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

This project is a unsupervised method for training a BART based model to convert sentences with idioms into sentences with their literal counterparts. In a nutshell, this is a stlylistic transformation between idiomatic sentences and literal sentences.  As in our paper, we refer to our method as BART-UCD.


As shown in the Figure below, the model architecture consists of three stages: (1) Embedding Stage, (2) Fusion Stage, and (2) Generation Stage. Please refer to our paper for more details. 

![modelacrchi](./images/unsupervised_framework_draft_v2.pdf  "The overview of the BART-UCD model architecture.")


### Built With

This model is heavily relying the resources/libraries list as following: 

* [PyTorch](https://pytorch.org/)
* [Huggingface's Transformer](https://huggingface.co/)
* [Sentence Transformer](https://www.sbert.net/)


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

All the dependencies for this project is listed in `requirements.txt`. You can install them via a standard command: 
```
pip install -r requirements.txt
```
It is highly recommanded to start a conda environment with PyTorch properly installed based on your hardward before install the other requirements. 



<!-- USAGE EXAMPLES -->
## Usage
Here, we show where the data is located and how to set up and run the training and inference for BART-UCD. 

### Data
To view an example of the expected training and testing data format, please see the files in `./data/sample_data'` Also, please refer to  `./src/utils/data_util.py` for the online data processing during training and inference for more clarity. 

The other files in the `./data` directory are the auxiliary data that the model or the data processing needs, such as the vocab of POS tags, and word definitions from different online dictionaries. 

### Training 
To train the model, 

- Edit `./meta_data/meta_data_sample_data.json` to set all the paths correctly. Currently, the paths are set to example data. 
- Edit `./config.py` to set experiment configurations. Set parameter `MODE = 'Train'` .
- In terminal and your envoirnment, run `python train.py`.

### Inference

- Edit `./meta_data/meta_data_sample_data.json` to set all the paths correctly. Currently, the paths are set to example data. 
- Edit `./config.py` to set experiment configurations. Set parameter `MODE = 'Test'` .
- In terminal and your envoirnment, run `python inference.py`.




<!-- LICENSE -->
## License

Distributed under the MIT License. 

The MIT License (MIT)

Copyright (c) 2021 Name

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



<!-- CONTACT -->
## Contact

Name Name - name@name.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements





<!-- File Structure -->
## File Structure
For clarity, the following is the file structure of this project. The main model files are located in `./src/model/`. The training and validation script for each epoch is in the file `./src/train_valid_test_step.py`. The top-level training and inference file are `./train.py` and `./inference.py`. All the model and experiment configurations are set in `./config.py`. 


BERT-UCD
├── checkpoints
├── config.py
├── data
│   ├── dictionaries
│   │   ├── dictionary_google.json
│   │   ├── dictionary_wiktionary.json
│   │   └── dictionary_wordnet.json
│   ├── pos_vocab
│   │   └── sample_POS_vocab.json
│   └── sample_data
│       ├── test_data.json
│       └── train_valid_data.json
├── inference.py
├── meta_data
│   └── meta_data_bart_sample.json
├── res
├── src
│   ├── __init__.py
│   ├── model
│   │   ├── attention.py
│   │   ├── highway_network.py
│   │   ├── __init__.py
│   │   ├── masked_conditional_generation_model_huggingface.py
│   ├── train_valid_test_step.py
│   └── utils
│       ├── data_util.py
│       ├── eval_util.py
│       ├── file_util.py
│       ├── __init__.py
│       ├── model_util.py
│       └── tensor_util.py
└── train.py
