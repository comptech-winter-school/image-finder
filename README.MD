# Image Finder CompTech2022

Image Finder is an implementation of the Image Finder project as part of the [CompTech2022 winter school](https://comptechschool.com/). 

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/app_gif.gif)

## About 

The product goal is possibility to help users to quickly finding the right images. 

#### Survey

According to a [survey](https://forms.gle/HmXznGKsynYdRoZ78) conducted as part of CompTech2022, users typically extract images on their devices themselves. This opportunity will be useful for people whose professional activities or hobbies are related to photo. In addition, we learned that this product will be useful to absolutely any user.

## The target audience

This product is aimed at all users.

## Demonstration

[Demonstration site](http://84.201.160.26:8503/)

## System operation

* Image search;  
* Search by text query (Russian/English); 
* Video search. 

## Structure
* 	`colabs` — directory with research experiments in colab notebooks;  
*  `test` – directory with tests;
* 	`assets` — directory with images;
* 	`main.py` — main file that includes all classes and functions for user-friendly web-service
* 	`faissindexer.py` — contains FAISS indexer that stores image embeddings and searches nearest neighbors for given text embedding;
* 	`dummyindexer.py` — contains simple indexer that stores image embbeddings and searches nearest neighbors for given text embedding by one-vs-all comparison;
* 	`hnsw_indexer.py` — contains HNSW indexer that stores image embbeddings and searches nearest neighbors for given text embedding by approximate nearest neighbor search;
* 	`embedder.py` — contains wrapper-classes for different CLIP models;
*  `searchmodel.py` — classes that connect indexers and CLIP embedders, load and store indexed images and their paths;
*  `CLIP_attention_maps.py` — attention maps for CLIP model;
*  `ruCLIP_attention_maps.py` — attention maps for RuCLIP model;
*  `requirements.txt` — list of dependencies 

## Principle of operation  

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/Untitled%20Diagram.drawio1.drawio-4.png)

For example
* Text query

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/Untitled%20Diagram.drawio-2.png)

* Image

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/Untitled%20Diagram.drawio1.drawio-3.png) 

## Usage 

This project was tested on python 3.7
1. Clone repository `git clone https://github.com/comptech-winter-school/image-finder.git`
2. Install required dependencies from requirements.txt.
3. Run command `streamlit run main.py --server.port {PORT}`
4. Copy IP-ADDRESS:PORT from terminal and paste it in browser
5. Select preferred indexer
6. Select text query or image method for processing
7. Select output image count
8. If you want to filter output results, you can use threshold slider
9. The images will be print with sorting of cosine distance.

## Docker Build

For running in Docker run these commands:

* `docker build -t streamlitapp:latest`
* `docker run -p 8501:8501 streamlitapp:latest`

App will be deployed at http://localhost:8501/
## Models

[CLIP](https://github.com/openai/CLIP) (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

[RuCLIP](https://github.com/sberbank-ai/ru-clip) (Russian Contrastive Language–Image Pretraining) is a multimodal model for obtaining images and text similarities and rearranging captions and pictures. RuCLIP builds on a large body of work on zero-shot transfer, computer vision, natural language processing and multimodal learning.

## ML experiments

### RuCLIP vs RuCLIP-SB

To solve the problem of image search by Russian-language queries, two models were considered: RuCLIP and RuCLIP-SB.

For the analysis, the CIFAR-100 dataset was selected, which contains 100 classes of 600 images with a size of 32x32. There are 500 training images and 100 test images in each class. This set is well suited for comparing models, as it has a wide variety of classes.

To evaluate the models, we will solve the classification problem. The definition of an image class can be considered as a search for an image by a query that is equal to the class label.

Precision, recall, accuracy, top-5 accuracy were analyzed. The RuCLIP model is better than RuCLIP-SB in all parameters, therefore, RuCLIP was chosen to solve the problem of image search by Russian-language queries.

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/RuClipVSRuClipSB.png)

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/Pre.png)

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/Recall.png)

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/Acc.png)

### RuCLIP vs RuCLIP-ONNX

The inference of RuCLIP and the optimized RuCLIP model via ONNX was investigated. Optimization gives a visible increase in speed.

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/Speed.png)

### Attention maps

Processing results for queries in the singular and plural are almost the same

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/1.png)

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/2.png)

Processing results depending on the number of iterations

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/3.png)

## Libraries 

* [pandas](https://github.com/pandas-dev/pandas) — software library in Python for data processing and analysis. 
* [numpy](https://github.com/numpy/numpy) — software library in Python that adds support for large multidimensional arrays and matrices. 
* [faiss](https://github.com/facebookresearch/faiss) — library of algorithms for finding nearest neighbors in linear space.
* [nmslib](https://github.com/nmslib/nmslib) — cross-platform similarity search library.
* [streamlit](https://github.com/streamlit/streamlit) — open-source app framework, the fastest way to build and share data apps.
* [torch](https://github.com/pytorch/pytorch) — open source deep learning framework.

## Team

* Developers: 
  * Anna Glushkova, 
  * Vasiliy Dronov, 
  * Kirill Keller, 
  * Alexandr Minin, 
  * Maxim Mashtakov, 
  * Vladislav Kuznetsov, 
  * Dmitry Moskalev
* Team Lead: 
  * Dmitry Moskalev
* Mentors:
  * Amir Uteuov, 
  * Vladimir Kilyazov.

