# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a CST-440 course repository containing projects and lab assignments focused on machine learning for embedded systems. The repository is organized with backend projects (Python-based data science/ML work) and lab assignment documentation.

## Course Context

### Topic 1: Introduction to Machine Learning on Microcontrollers (Jan 5-25, 2026)

This topic covers the transition from machine learning on large computing devices (laptops, smartphones) to constrained embedded devices (microcontrollers). The focus is on designing, training, and deploying machine learning models that can run on resource-limited hardware.

**Learning Objectives:**
- Design a deep learning workflow for an embedded device
- Build and train a machine learning model
- Deploy a machine learning application on a microcontroller

**Key Project: CLC – Machine Learning on a Microcontroller**

The main collaborative project involves designing and training a machine learning model to compute trigonometric functions, then deploying it to a microcontroller. The workflow follows these steps:

1. **Model Design** - Create an architecture that fits the task and resource constraints
2. **Model Building** - Implement the model using frameworks like TensorFlow
3. **Model Training** - Teach the model using collected/preprocessed data
4. **Application Building** - Package the model into a deployable application
5. **Testing** - Validate accuracy and efficiency of the model
6. **Deployment** - Deploy to microcontroller (e.g., Arduino) and capture output

**Lab Questions Focus:**
- Training data design for mathematical functions
- Neural network implementation using TensorFlow
- Deployment workflows and diagrams
- Differences between microcontroller vs. desktop deployment

## Project Structure

```
CST-440/
├── backend/
│   └── Project1/          # Python-based data science/ML project
│       └── requirements.txt
├── Lab Questions/         # Lab assignment documentation
└── README.md
```

## Development Setup

### Backend/Project1 (Python Data Science)

1. Navigate to the project directory:
   ```sh
   cd backend/Project1
   ```

2. Create and activate virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Dependencies

Project1 uses the following Python packages:
- **Data manipulation**: numpy, pandas
- **Machine learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Scientific computing**: scipy
- **Interactive development**: jupyter, ipython
- **Utilities**: requests, python-dotenv

## Architecture Notes

### Backend Structure

The backend is organized by project number (Project1, etc.). Each project is self-contained with its own virtual environment and requirements. Projects typically follow the machine learning workflow:

- **Data Collection & Preprocessing** - Gathering and normalizing training data
- **Model Design & Training** - Using TensorFlow/scikit-learn to build and train models
- **Model Optimization** - Preparing models for deployment on constrained devices
- **Deployment Preparation** - Converting models for microcontroller compatibility

### Lab Questions

The `Lab Questions/` directory contains text-based documentation and answers for course lab assignments. These are standalone documents separate from the implementation projects, focusing on conceptual understanding of ML deployment workflows, data design, and embedded systems constraints.

## Git Workflow

The main branch is `main`. The repository excludes:
- Virtual environments (`venv/`, `*/venv/`)
- Personal projects (`PersonalProjects/`)
