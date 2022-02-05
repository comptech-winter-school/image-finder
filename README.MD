# Image Finder CompTech2022

ImgFinder is an implementation of the Image Finder project as part of the [CompTech2022 winter school] (https://comptechschool.com/). 

## Structure

<!--main.py - main file -->
* 	`faiss.py` — contains FAISS indexer that stores image embeddings and searches nearest neighbors for given text embedding;
* 	`embedder.py` — contains wrapper-classes for CLIP models;
* 	`colabs/comptech_ruclip_usage.ipynb` — answering model;  
* 	`assets` — img files. 

## About 

The idea of this product is to possibly help users in the task of quickly finding the right images. This opportunity will be useful for people whose professional activities or hobbies are related to photo. 

## Principle of operation  

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/Untitled%20Diagram.drawio.png)

For example 

![](https://github.com/comptech-winter-school/image-finder/blob/main/assets/Untitled%20Diagram.drawio-2.png)

## The target audience

This product is aimed at all users. 

## Usage

## Library 

* Pandas — software library in Python for data processing and analysis. 
* NumPy — software library in Python that adds support for large multidimensional arrays and matrices. 
* Faiss — library of algorithms for finding nearest neighbors in linear space. 

## System operation

* Image search;  
* Search by text query (Russian/English); 
* Video search. 

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

