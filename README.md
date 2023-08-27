# **State-of-the-Art Embedding Generation with GTE and BGE Models**

Welcome to the project that demonstrates the generation of high-quality embeddings using state-of-the-art Global Text Embeddings (GTE) and Binary Global Embeddings (BGE) models from the Hugging Face Transformers library. This project showcases the utilization of these models through a FastAPI-based application that allows you to easily generate embeddings for your text data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Endpoints](#endpoints)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to provide an efficient and user-friendly interface for generating contextual embeddings from text data using advanced language models. By leveraging the power of GTE and BGE models, the application allows users to quickly obtain embeddings that capture the semantic content of their input text.

## Features

- Generate embeddings using GTE and BGE models.
- FastAPI-based API endpoints for easy integration.
- Support for different model variants.
- Efficient and scalable embedding generation.

## Setup

### Installation

1. Clone this repository to your local machine.
2. Create a virtual environment (recommended).
3. Install the required dependencies using the following command:

   ```bash
   pip install transformers sentence-transformers
   ```

### Running the Application

1. Start the FastAPI application using the following command:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080
   ```

   This will start the application on http://localhost:8080.

## Endpoints

The application provides the following endpoints for embedding generation:

- `/base`: Generate embeddings using the GTE base model.  [768 dimension]
- `/large`: Generate embeddings using the GTE large model. [1024 dimension]
- `/bgelarge`: Generate embeddings using the BGE large model. [1024 dimension]
- `/e5_large_v2`: Generate embeddings using the e5_large_v2 model. [1024 dimension]

## Usage

1. Make a POST request to the desired endpoint (e.g., `/base`) with JSON data containing the `"text"` key and the text data for which you want to generate embeddings. For example:

   ```json
   {
       "text": "This is an example text for embedding generation."
   }
   ```

2. The response will contain the generated embeddings in JSON format.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to create a pull request or open an issue.
