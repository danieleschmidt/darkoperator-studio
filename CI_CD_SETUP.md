# CI/CD Pipeline Setup Guide

Since GitHub Actions workflow creation requires additional permissions, here's how to manually set up the CI/CD pipeline:

## Option 1: GitHub Actions (Manual Setup)

Create `.github/workflows/ci-cd.yml` manually in the GitHub repository with the following content:

```yaml
name: DarkOperator CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        python -m pytest tests/ --cov=darkoperator --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run security scan
      uses: pypa/gh-action-pip-audit@v1.0.8

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: |
        docker build -t darkoperator:${{ github.sha }} .
        docker build -t darkoperator:latest .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push darkoperator:${{ github.sha }}
        docker push darkoperator:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to production
      run: |
        # Add your deployment commands here
        echo "Deploying DarkOperator to production..."
        # kubectl apply -f k8s/
```

## Option 2: GitLab CI/CD

Create `.gitlab-ci.yml`:

```yaml
stages:
  - test
  - security
  - build
  - deploy

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

test:
  stage: test
  image: python:3.11
  parallel:
    matrix:
      - PYTHON_VERSION: ["3.9", "3.10", "3.11", "3.12"]
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - python -m pytest tests/ --cov=darkoperator

security:
  stage: security
  image: python:3.11
  script:
    - pip install pip-audit
    - pip-audit

build:
  stage: build
  image: docker:20.10.16
  services:
    - docker:20.10.16-dind
  script:
    - docker build -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE
  only:
    - main

deploy:
  stage: deploy
  image: kubectl:latest
  script:
    - kubectl apply -f k8s/
  only:
    - main
```

## Option 3: Jenkins Pipeline

Create `Jenkinsfile`:

```groovy
pipeline {
    agent any
    
    stages {
        stage('Test') {
            parallel {
                stage('Python 3.9') {
                    steps {
                        sh 'python3.9 -m pytest tests/'
                    }
                }
                stage('Python 3.11') {
                    steps {
                        sh 'python3.11 -m pytest tests/'
                    }
                }
            }
        }
        
        stage('Security Scan') {
            steps {
                sh 'pip-audit'
            }
        }
        
        stage('Build') {
            when { branch 'main' }
            steps {
                sh 'docker build -t darkoperator:${BUILD_NUMBER} .'
                sh 'docker tag darkoperator:${BUILD_NUMBER} darkoperator:latest'
            }
        }
        
        stage('Deploy') {
            when { branch 'main' }
            steps {
                sh 'kubectl apply -f k8s/'
            }
        }
    }
}
```

## Setup Instructions

1. Choose your preferred CI/CD platform
2. Copy the appropriate configuration file to your repository
3. Configure secrets/variables:
   - `DOCKER_USERNAME` and `DOCKER_PASSWORD` for Docker registry
   - Kubernetes credentials for deployment
4. Enable the pipeline in your CI/CD platform
5. Test with a small commit to trigger the pipeline

The pipeline will automatically:
- Run tests across multiple Python versions
- Perform security scanning
- Build Docker images on main branch
- Deploy to production environment