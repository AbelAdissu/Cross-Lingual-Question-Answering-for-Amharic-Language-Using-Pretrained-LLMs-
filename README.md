## **INTRODUCTION** 📖 
Welcome to the Amharic Text Generation project, a journey into the realm of natural language processing for low-resource languages. This initiative taps into the power of large language models, particularly transformer-based architectures, to understand and generate the Amharic language, which historically had limited exposure in the digital world of AI. Our project distinguishes itself by meticulously constructing transformer models from scratch and adapting pre-trained large language models to grasp the syntactic and semantic nuances of Amharic. 

This project is unique in its technical approach: we've built a Bigram Language Model and a more complex GPT (Generative Pre-trained Transformer) model from the ground up, specifically tailored for Amharic. The Bigram Model serves as a fundamental step into language modeling, focusing on the prediction of the next character in a sequence based on its predecessor, thereby grasping the basic language patterns. On the other hand, our GPT model, which is the centerpiece of this project, employs a sophisticated multi-head attention mechanism, enabling the model to simultaneously process different parts of the input sequence and learn a richer understanding of context and relationships between words.

We've pushed the boundaries further by incorporating advanced transformer architectures in our GPT model. These architectures rely on stacks of decoders—a powerful component that allows the model to generate coherent and contextually relevant text.

Additionally, we've explored and adapted various scales of pre-trained models, including:
* 'gpt2' with 124M parameters
* 'gpt2-medium' with 350M parameters
* 'gpt2-large' with 774M parameters
* 'gpt2-xl' with 1558M parameters

Each of these models offers a different level of complexity and learning capacity, allowing us to experiment and find the most effective approach for the Amharic language. This range of models helps in understanding the scalability and adaptability of our approach to the unique challenges presented by Amharic text generation.

## **TABLE OF CONTENTS** 📑
* **Getting Started**
   * Prerequisites
   * Tools and Libraries
   * Installation
   * Dataset
* **Project Pipeline**
   * Data Preprocessing
   * Model Development
   * Training
   * Evaluation and Testing
   * Text Generation
* **Future Work**
   *  In-depth Research
   *  Exploring Other Datasets
   *  Cross-lingual NLP
   *  Hyper-parameter Tuning and Pre-trained Models
   *  Reviewing Performance Enhancement Techniques
* **Contact**
* **Acknowledgements**

## **GETTING STARTED** 🚀 
### 🔧 **Prerequisites** 
    
    * Python 3.8 or higher
    
    * pip (latest version)
    
    * Git
    
    * Virtual Environment (optional, recommended)
    
    * PyTorch (specify the minimum version required, e.g., 1.7.0)
    
    * CUDA Toolkit (if leveraging GPU acceleration)
    
    * Transformers Library (from Hugging Face)
    
### 🧰 **Additional Tools and Libraries** 

    * NumPy (for numerical operations)
    
    * Pandas (for data manipulation and analysis)
    
    * Matplotlib (optional, for plotting and visualization)
    
    * Jupyter Notebook (optional, for interactive development and testing)
    
    * tqdm (for progress bars in loops)
    
### ⚙️ **Installation**

Clone the repository: 
https://github.com/quinbez/Large_Language_Models_For_Low_Resource_Languages.git

### 📊 **Dataset**
The dataset used for this project can be found at the following link:
https://github.com/leobitz/amharic_word_embeddings

##  PROJECT PIPELINE 🔀
### 1. 🔍 **Data Preprocessing**
* **Cleaning**: Removal of noise and irrelevant content from the raw Amharic dataset.
* **Normalization**: Standardization of text, handling variations in characters and script.
* **Character-Level Tokenization**: Breaking down text into a sequence of characters.
### 2. 🧠 **Model Development**
* **Building Custom Transformer**: Creating transformer models from scratch tailored for Amharic text.
* **Utilizing Pre-trained Models**: Adapting existing large language models to understand Amharic nuances.
### 3. 🏋️ **Training**
* **Feeding Data**: Inputting the preprocessed, tokenized data into the model.
* **Optimization**: Tuning parameters and refining the model's ability to generate coherent text.
### 4. 🔬 **Evaluation and Testing**
* **Loss Estimation**: Regular assessment of model performance during training.
* **Quality Checks**: Validating the model's output for accuracy and linguistic consistency.
### 5. ✍️ **Text Generation**
* **Seed Input**: Starting the generation process with an initial text (seed).
* **Iterative Generation**: Producing new text, character by character, based on the learned patterns.

## **FUTURE WORK** 🔮 
This project aims to enhance the accuracy of Amharic text generation, and there are several avenues we plan to explore for future improvements. These areas of focus include:

1. **In-depth Research**: Conducting a comprehensive analysis of existing literature and research papers related to Amharic text generation to gain deeper insights and inform our approach.

2. **Exploring Other Datasets**: Investigating additional datasets for Amharic language that can complement or enhance the existing dataset used in our project, thereby capturing a wider range of language patterns.

3. **Cross-lingual NLP**: Exploring cross-lingual natural language processing techniques to leverage knowledge from other languages and applying it to Amharic for improved accuracy.

4. **Hyper-parameter Tuning and Pre-trained Models**: Experimenting with different hyper-parameters and pre-trained models specifically designed for low-resource languages to optimize the performance of our text generation models.

5. **Reviewing Performance Enhancement Techniques**: Keeping up with the latest research advancements and exploring techniques to further enhance the accuracy of fine-tuned large language models for low-resource languages.



### 📂 **Project Link** 

https://github.com/quinbez/Large_Language_Models_For_Low_Resource_Languages/blob/main/Large_Language_Models_For_Low_Resource_Languages.ipynb

## **ACKNOWLEDGEMENTS** 🙏
1. Andrej Karpathy - Chief Scientist at OpenAI and a leading figure in the field of deep learning and computer vision.
2. Andrew Ng - Co-founder of Coursera, former Chief Scientist at Baidu, and renowned AI researcher and educator.
3. Christopher Manning - Professor of Computer Science and Linguistics at Stanford University, specializing in natural language processing and computational linguistics.
