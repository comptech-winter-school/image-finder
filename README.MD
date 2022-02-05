# Image Finder CompTech2022

ImgFinder is an implementation of the Image Finder project as part of the [CompTech2022 winter school](https://comptechschool.com/). 

## About 

The product goal is possibility to help users in the task of quickly finding the right images. This opportunity will be useful for people whose professional activities or hobbies are related to photo. 

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
* 	`main.py` - main file that includes all classes and functions for user-friendly web-service
* 	`faissindexer.py` — contains FAISS indexer that stores image embeddings and searches nearest neighbors for given text embedding;
* 	`dummyindexer.py` — contains simple indexer that stores image embbeddings and searches nearest neighbors for given text embedding by one-vs-all comparison;
* 	`embedder.py` — contains wrapper-classes for different CLIP models;
*  `searchmodel.py` — classes that connect indexers and CLIP embedders, load and stores indexed images and their paths;
*  `CLIP_attention_maps.py` — attention maps for CLIP model;
*  `ruCLIP_attention_maps.py` — attention maps for ruCLIP model;
*  `requirements.txt` — list of dependencies 

## Principle of operation  

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/Untitled%20Diagram.drawio1.drawio-4.png)

For example

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/Untitled%20Diagram.drawio-2.png)


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


## Libraries 

* Pandas — software library in Python for data processing and analysis. 
* NumPy — software library in Python that adds support for large multidimensional arrays and matrices. 
* Faiss — library of algorithms for finding nearest neighbors in linear space. 
* Glob - module that finds all paths that match a given pattern according to the rules used by the Unix shell.
* Math - module that provides access to some popular mathematical functions and constants.
* abc - module that helps design abstract entities.

## Some screenshots of Streamlit main page
![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/streamlit_1.png)
![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/streamlit_2.png)

## Team

* Developers: 
  * Anna Glushkova, 
  * Vasiliy Dronov, 
  * Vadim Kozlov, 
  * Kirill Keller, 
  * Alexandr Minin, 
  * Maxim Mashtakov, 
  * Vladislav Kuznetsov, 
  * Dmitry Moskalev
* Team lead: 
  * Dmitry Moskalev
* Mentors:
  * Amir Uteuov, 
  * Vladimir Kilyazov.

