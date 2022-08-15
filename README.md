# Introduction

This repo is a complete demo of real-time model monitoring using Evidently. Within the repo, you will find:

* A script to download and process the training data
* A model training pipeline to train a simple regression model that predicts house prices.
* A model server that exposes our house price model through a REST API.
* Some scripts that simulate different scenarios: e.g. model drift vs normal input.
* An Evidently model monitoring service which collects inputs and predictions from the model and computes metrics such as drift.
* A monitoring dashboard which uses Prometheus and Grafana to visualise in real-time the monitoring metrics.
* Docker Compose to run the whole thing.


